from functools import partial
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pL
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info, rank_zero_warn
from deep_gwbse.from_model.e2vae import EquivariantVAE, kl_loss_mean
from deep_gwbse.from_model.e3vae import EquivariantVAE3D
from torch.utils.data import DataLoader
from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.trainer import TrainDataModule
from deep_gwbse.from_model.vaetrainer import wfn_collate_fn, wfn_3d_wigner_collate_fn
from typing import Callable, Type


class WFNTrainModule(pL.LightningModule):
    '''(LightningModule is a child class of torch.nn.Module.)'''
    def __init__(self, model: torch.nn.Module,
                 coeff_mse=1, coeff_kl=0.02,
                 lr=2e-3, lr_decay=0.5, lr_patience=400, lr_monitor="val/total_loss",
                ):
        '''
        LightningModule for VAE training.
        Args:
            model: the VAE model to be trained. It should be called as `x_recon, mu, logvar = model(x)`.
            coeff_mse: coefficient for mse loss in Loss. MSE Ignored if None or <= 0.
            coeff_kl: coefficient for kl loss in Loss. KL loss ignored if None or <= 0.
            lr: learning rate.
            lr_decay: learning rate decay factor for ReduceLROnPlateau scheduler.\
                lr*=lr_decay if no improvement on lr_monitor after lr_patience epochs.
            lr_patience: patience for ReduceLROnPlateau scheduler.
            lr_monitor: metric name to be monitored for ReduceLROnPlateau scheduler.
        '''
        super().__init__()
        
        self.model = model
        self.coeff_mse = coeff_mse
        self.coeff_kl = coeff_kl
        
        if coeff_kl is not None and coeff_kl > 0.:
            self.kl_loss = kl_loss_mean
        
        if self.coeff_mse is not None and self.coeff_mse > 0.:
            self.mse = torch.nn.MSELoss(reduction="mean")
        
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
    
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, batch, batch_idx, mode="train"):
        x, mask = batch
        x_recon, mu, logvar = self.model(x)
        mask = ~mask
        total_loss = torch.tensor(0.0, device=self.device)
        loss_dict = {}
        # mse loss
        if self.coeff_mse is not None and self.coeff_mse > 0.:
            # x_recon_masked = x_recon[:, mask.squeeze()]
            # x_masked = x[:, mask.squeeze()]
            # mse_loss = self.mse(x_recon_masked, x_masked)
            mse_loss = self.mse(x_recon*mask, x*mask)
            loss_dict["mse_loss"] = mse_loss
            total_loss += self.coeff_mse * mse_loss
        # kl loss
        if self.coeff_kl is not None and self.coeff_kl > 0.:
            kl = self.kl_loss(mu, logvar)
            loss_dict["kl"] = kl
            total_loss += self.coeff_kl * kl
       
        loss_dict["total_loss"] = total_loss

        # log
        for k, v in loss_dict.items():
            self.log(f"{mode}/{k}", v, prog_bar=(k == "total_loss"), on_step=False, on_epoch=True,
                     batch_size=x.shape[0], sync_dist=True)

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.calculate_loss(batch, batch_idx, mode="train")
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.calculate_loss(batch, batch_idx, mode="val")
        return loss_dict["total_loss"]
    
    def test_step(self, batch, batch_idx):
        loss_dict = self.calculate_loss(batch, batch_idx, mode="test")
        return loss_dict["total_loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]


class SubBatchWFNData(Dataset):
    def __init__(self, mat_idx, wfdata, slice_list):
        '''
        Create a sub-batched dataset. self[i] is defined by `orig_idx, start, end = slice_list[i]`, `self[i] = wfdata[mat_idx.index(orig_idx)][start: end]`.
        In the multi-gpu case, `mat_idx` is the indices of materials used by this process, and `wfdata` is the corresponding data to `mat_idx` on this process.
        Args:
            mat_idx (list): list of material indices corresponding to wfdata.
            wfdata (list): list of wfdata arrays of the same length as mat_idx, each of shape (nk*nb, ...), same len as mat_idx.
            slice_list (list): list of (mat_idx, start, end) defining the slices of each new data point.
        
        Examples:
            If this process only uses material 3 and 4, then an example is:
            >>> SubBatchWFNData(mat_idx = [3, 4], wfdata = [wfdata_3, wfdata_4], slice_list = [(3, 0, 10), (3, 10, 20), (4, 0, 15)])
        '''
        super().__init__()
        assert len(mat_idx) == len(wfdata)
        self.slice_list = slice_list
        self.wfdata = {mat_idx[i]: wfdata[i] for i in range(len(mat_idx))}

    def __len__(self):
        return len(self.slice_list)

    def __getitem__(self, index):
        orig_idx, start, end = self.slice_list[index]
        return self.wfdata[orig_idx][start:end]


class WFNDataModule(TrainDataModule):
    '''
    This module splits an original material (nk, nb, ...) datapoint into several points (N1, ...), (N2, ...), ...
    We do the splitting instead of creating a new dataset with each wavefunction as a data point and then collating with batch > 1
    because the former is more convenient for each process to load the data it uses.
    '''
    def __init__(self, base_dataset: ManyBodyData, max_batch_size: int = 64, global_stage="train",
                 train_stage_split=[0.8, 0.1, 0.1], collate_fn=None, from_GW_data=False):
        '''
        This Datamodule splits an original datapoint (nk, nb, ...) in base_dataset to (N_1, ...), (N_2, ...), ... with N_i <= max_batch_size.\
        Indexing `sub_batch_data[i]` directly gives a np array.
        Args:
            base_dataset: the original dataset containing wfdata. Each data point is expected to have a key "wfn" or "src/wfn" (if from_GW_data=True) containing the wfdata array of shape (nk, nb, ...).
            max_batch_size: the maximum batch size for each sub-batch.
            global_stage: "train", "val", or "test". The stage determines which part of the base_dataset is used.
            train_stage_split: list of 3 floats, defining the split of base_dataset into train/val/test sets.
            collate_fn: collate function for DataLoader. Each function should accepts wfn of shape (N_sub, ...) instead of (nk, nb, ...).
            from_GW_data: whether the data is from GW dataset. If True, wfn is accessed via "src/wfn", otherwise "wfn".
        '''
        super().__init__(dataset=None, global_stage=global_stage, train_stage_split=train_stage_split, batch_size=1, collate_fn=collate_fn)
        self._base_dataset = base_dataset
        self._slice_list = []  # (orig_idx, start, end), each elem is a data point in the sub-batched dataset, len = len(sub-batched datset).
        self._orig2new_idx_map = [[] for _ in range(len(base_dataset))]  # _orig2new_idx_map[orig_idx] == [new_idx's in sub-batched dataset]
        self.orig_len = len(base_dataset)
        if from_GW_data:
            self.fetch_str = "src/wfn"
        else:
            self.fetch_str = "wfn"

        # create slice list: each elem is [orig_idx, start, end]
        new_idx = 0  # idx in new sub-batched dataset
        sub_batch_size_all = []
        for mat_idx, data in enumerate(base_dataset):
            keys = self.fetch_str.split("/")
            for k in keys:
                data = data[k]
            N_batch_orig = data.shape[0] * data.shape[1]  # nk * nb
            num_sub_batch = int(np.ceil(N_batch_orig / max_batch_size))
            sub_batch_size_base = N_batch_orig // num_sub_batch
            remainder = N_batch_orig % num_sub_batch
            sub_batch_sizes = [sub_batch_size_base + 1] * remainder + [sub_batch_size_base] * (num_sub_batch - remainder)
            sub_batch_size_all.extend(sub_batch_sizes)
            orig_batch_idx_start = 0  # batch idx in original dataset
            for s in sub_batch_sizes:
                self._slice_list.append([mat_idx, orig_batch_idx_start, orig_batch_idx_start+s])
                orig_batch_idx_start += s
                self._orig2new_idx_map[mat_idx].append(new_idx)
                new_idx += 1
        if max(sub_batch_size_all) - min(sub_batch_size_all) > 10:
            rank_zero_warn("Difference between max batch size and min batch size is greater than 10.")

    def merge(self, input_list):
        '''Merge a sub_batched list into original index.

        Args:
            input_list (iterable): an iterable of arrays with length == len(self), and input_list[i].shape[0] == self[i].shape[0].

        Returns:
            out_list (list): a list of length == self.orig_len. out_list[i] = np.concatenate(input_list[self._orig2new_idx_map[i]]).
        '''
        if len(input_list) != len(self._slice_list):
            raise ValueError("Input list length does not match dataset length!")
        out_list = []
        for i in range(self.orig_len):
            out_list.append(np.concatenate([input_list[new_idx] for new_idx in self._orig2new_idx_map[i]], axis=0))
        return out_list

    def setup(self, stage):
        # initiate data on this process
        indices_p = self._split_process(len(self._slice_list))  # indices for this process
        slice_list_p = [self._slice_list[i] for i in indices_p]  # slice list for this process
        # load data for each process
        mat_indices_p = np.unique([self._slice_list[i][0] for i in indices_p])  # mat idx used in this process
        # load wfdata (nk*nb, H, W, Z, 2) involved in this process
        dataset_name = os.path.join(self._base_dataset.dataset_dir, self._base_dataset.dataset_fname)
        wfdata_p = []  # wfn arrays used in this process
        for mat_idx in mat_indices_p:
            wfn = self._base_dataset.datapoint_interface_h5(dataset_name, self._base_dataset.info.mat_id[mat_idx], mode='r')
            for k in self.fetch_str.split("/"):
                wfn = wfn[k]
            # if np.iscomplexobj(wfn):
                # wfn = np.stack([wfn.real, wfn.imag], axis=-1)
            nk, nb = wfn.shape[:2]
            wfdata_p.append(wfn.reshape(nk*nb, *wfn.shape[2:]))  # reshape to (nk*nb, ...)
            # nk, nb, H, W, C, _ = wfn.shape
            # wfdata_p.append(wfn.reshape(nk*nb, H, W, C, 2))

        self.dataset = SubBatchWFNData(mat_indices_p, wfdata_p, slice_list_p)

        super().setup(stage)


def init_input_channels_real(config, wfdata: ManyBodyData):
    from_GW_data = not (wfdata.info.dataset_type.lower() == "wfn")
    use_3d = config["dataset"]["use_3d"]
    if not use_3d:
        if from_GW_data:
            wfn_0 = wfdata[0]["src"]["wfn"]
        else:
            wfn_0 = wfdata[0]["wfn"]
        input_channels = wfn_0.shape[-1]  # Z dimension, do not use cell_slab_truncation for better compatibility
    else:
        input_channels = 1
    return input_channels
 

def init_collate_real(config):
    if config["dataset"]["use_3d"]:
        collate_fn = partial(wfn_3d_wigner_collate_fn, batched_data=True)
    else:
        collate_fn = partial(wfn_collate_fn, batched_data=True)
    return collate_fn


def init_model_real(config, input_channels):
    model_config = config["model"]
    model_config["input_channels"] = input_channels
 
    net = model_config.pop("net", None)
    if net is None:  # determine automatically
        net = "equivariantvae" if not config["dataset"]["use_3d"] else "equivariantvae3d"
    else:
        net = net.lower()

    if net == "equivariantvae":
        vae = EquivariantVAE(**model_config)
    elif net == "equivariantvae3d":
        vae = EquivariantVAE3D(**model_config)
    else:
        raise ValueError(f"Unknown net type: {net}")
    
    if net == "equivariantvae3d":
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.enabled = True
    
    return vae


def run_vae(config, init_input_channels:Callable, init_collate:Callable, init_model:Callable, TrainClass: Type=WFNTrainModule):  # the main function running training and testing.
    '''
    This implements the main logic of I/O and training / testing.

    Args
    -------
        config: dict
            The input config dict, directly loaded from yaml.
        init_input_channels: function(config, wfdata) -> int
            A function determining input_channels from wfdata. `wfdata` is an instance of `ManyBodyData`.
        init_collate: function(config) -> callable
            A function determining collate_fn for DataLoader from config.
        init_model: function(config, input_channels) -> torch.nn.Module
            A function initializing the model from config and input_channels.
        TrainClass:  Type
            LightningModule class for training. It should be a child class of `pL.LightningModule`. Pass the type, not an instance.
    '''
    # global setting
    seed = config["system"]["seed"]
    precision = config["system"]["precision"]

    if seed is not None:
        pL.seed_everything(seed, workers=True)

    # initiate dataset (lazy load)
    dataset_slice = config["dataset"]["dataset_slice"]
    if dataset_slice is not None:
        dataset_slice = slice(dataset_slice[0], dataset_slice[1], dataset_slice[2] if len(dataset_slice) == 3 else 1)
    wfdata = ManyBodyData.from_existing_dataset(config["dataset"]["dataset_dir"], data_slice=dataset_slice, lazy_load=True)
    if len(wfdata) == 0:
        raise RuntimeError("No data in dataset!")
    from_GW_data = not (wfdata.info.dataset_type.lower() == "wfn")
    input_channels = init_input_channels(config, wfdata)
   
    # collate functions
    collate_fn = init_collate(config)

    # create datamodule
    dataset_param = {"base_dataset": wfdata,  "max_batch_size": config["dataset"]["max_batch_size"],
                     "global_stage": config["system"]["stage"], "collate_fn": collate_fn,
                     "from_GW_data": from_GW_data,
                     "train_stage_split": config["dataset"]["train_stage_split"]}
    data_lightning = WFNDataModule(**dataset_param)

    # model
    rank_zero_info("Building VAE model with the following hyperparameters:")
    vae = init_model(config, input_channels)

    # lightning module and trainer
    conf_opt = config["optimization"]
    lightning_params = {"model": vae,
                        "lr": conf_opt["lr"], "lr_decay": conf_opt["lr_decay"],
                        "lr_patience": conf_opt["lr_patience"], "lr_monitor": conf_opt["monitor"],
                        "coeff_mse": conf_opt["coeff_mse"], "coeff_kl": conf_opt["coeff_kl"],
                        }

    # train stage
    if config["system"]["stage"].lower() == "train":
        # lightning_params["save_test_result"] = False
        # trainer
        trainer_params = {"accelerator": config["system"]["accelerator"],
                          "num_nodes": config["system"].get("num_nodes", 1),
                          "devices": config["system"]["devices"],
                          "precision": "64-true" if precision == 64 else "32-true",
                          "default_root_dir": config["system"]["train_dir"],
                          "min_epochs": config["optimization"]["min_epochs"],
                          "max_epochs": config["optimization"]["max_epochs"],
                          "gradient_clip_val": config["optimization"]["gradient_clip_val"],
                          "deterministic": "warn" if config["system"]["seed"] is not None else False,
                          }
        if config["system"]["devices"] >= 2:
            trainer_params["strategy"] = "ddp"
            trainer_params["use_distributed_sampler"] = False
        if config["optimization"]["monitor_time"]:
            trainer_params["profiler"] = "simple"
        # callbacks
        callbacks = [
            LearningRateMonitor(),  # log lr
            EarlyStopping(  # early stopping
                monitor=config["optimization"]["monitor"],
                patience=config["optimization"]["train_patience"],
                min_delta=1e-6,  # change greater than min_delta is considered an improvement
                mode="min",
                check_finite=True,
            ),
            ModelCheckpoint(  # save best model
                filename="best-epoch-{epoch}",
                save_top_k=1,
                verbose=False,
                monitor=config["optimization"]["monitor"],
                mode='min',
                every_n_epochs=1,
            )
        ]
        trainer_params["callbacks"] = callbacks

        if not config["system"]["resume"]:  # train from scratch
            if config["system"]["load_from_ckpt"]:
                # load from checkpoint as initial values, not resume training
                vae_lightning = TrainClass.load_from_checkpoint(config["system"]["ckpt_dir"], **lightning_params)
                rank_zero_info(f"Load from checkpoint: {config['system']['ckpt_dir']}")
            else:
                # initialize from scratch
                vae_lightning = TrainClass(**lightning_params)
            trainer = pL.Trainer(**trainer_params)
            rank_zero_info("Start training.")
            trainer.fit(vae_lightning, datamodule=data_lightning)
            rank_zero_info("Training finished.")

        else:  # resume from ckpt
            rank_zero_info("Resume from checkpoint:", config["system"]["ckpt_dir"])
            vae_lightning = TrainClass(**lightning_params)
            trainer = pL.Trainer(**trainer_params)
            rank_zero_info("Start training.")
            trainer.fit(vae_lightning, datamodule=data_lightning, ckpt_path=config["system"]["ckpt_dir"])
            rank_zero_info("Training finished.")

        if config["dataset"]["train_stage_split"][2] > 0:
            rank_zero_info("Start testing.",)
            trainer.test(vae_lightning, datamodule=data_lightning)
            rank_zero_info("Testing finished.")

    elif config["system"]["stage"].lower() == "test":
        assert config["system"]["ckpt_dir"] is not None, "ckpt_dir is required for testing."
        save_test_results = config["dataset"].get("save_test_results", False)
        lightning_params["save_test_result"] = save_test_results
        vae_lightning = TrainClass.load_from_checkpoint(config["system"]["ckpt_dir"], **lightning_params)
        # vae_lightning.model.AE_switch(True)  # set to AE mode
        # vae_lightning.coeff_kl = None  # do not evaluate kl
        rank_zero_info(f"Load from checkpoint: {config['system']['ckpt_dir']}")
        # test only trainer
        trainer_params = {"accelerator": config["system"]["accelerator"], "devices": 1,
                          "precision": "64-true" if precision == 64 else "32-true",
                          "default_root_dir": config["system"]["train_dir"],
                          "deterministic": (config["system"]["seed"] is not None),
                          "strategy": "auto",
                          }
        trainer = pL.Trainer(**trainer_params)
        rank_zero_info("Start testing.")
        trainer.test(vae_lightning, datamodule=data_lightning)
        rank_zero_info("Testing finished.")

    rank_zero_info("DONE.")


def main(config):
    run_vae(config, init_input_channels_real, init_collate_real, init_model_real)