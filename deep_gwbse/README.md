## General workflow
```

Molecule Dynamncse─┐
      external src───(Collect)─>fp-input ──(QE/SIESTA/BGW + HPRO)─>─┌── ml-train-set──(ML)─> model
               ...─┘                                                └── ml-test-set
```
**Path 1** Tweisted-angle study of hBN (**not working well**)
```

1. Train:
supercell.cif─(flow.py)─> MD─(md.py)─>fp-input─(flows.py)─> ml_dataset ──(deep-collect.py, deephe3-train.py)─> model

2. Use:
twist.cif─┌──(deephe3-xx.py, diag_plot.py)─> band.png 
    model─┘
```

**Path 2** MBFormer for GW-BSE
```
--Path 2--:
1. Train:
external database─>stru_input─(flows.py,flows-aug.py)─>flows─>(data.py)─>dataset.h5─(trainer.py)─> model

2. Use:
Features: G0W0, BSE (binding energy, |<cvk|S>|)
```

## Folder Structure

### 1. **stru-input** folder
The stru-input folder contains the crystal structures
```bash
stru-input
├── mat-1 # (extensible)
|   └── stru.cif
├── mat-2
|   └── stru.cif
└── ...
```
Related files on top of the folder:
- `flow.py` (**unit-test**): `-c` reads .json file, create simple material flow.
- `flows.py` (**unit-test**): `-c` reads .json file, create multiple material flows.
- `flows-augmentations.py`: `-c` reads .json file, create `GW` or `BSE` augmentation flows for finished flows.
- `fptask.py`: customized task for the `flow.py` script.
- `collect_tool.py`:
    - md: `collect_tool.py md -md_input MD_INPUT -md_output MD_OUTPUT -md_suffix MD_SUFFIX`
- `config/single_mat_config.json`: The configuration file for the `flow.py`(single material flow).
- `config/fpconfig.json`: The configuration file for the `flows.py` script(multiple material flows).

### 2. **pp** folder
The pp folder contains all .upf and .psml for QE and SIESTA
```
pseudo_src/ # (built-in)
├── ele1.upf
├── ele2.upf
├── ...
├── ele1.psf/psml
├── ele2.psf/psml
└── ...
```

### 3. **flows** folder
```bash
flows/
├── mat-1
|   ├── config.json
|   ├── stru.cif
|   ├── pp/ # (built-in)
|   |   ├── ele1.upf
|   |   ├── ele2.upf
|   |   ├── ...
|   |   ├── ele1.psf/psml
|   |   ├── ele2.psf/psml
|   |   └── ...
|   ├──01-density
|   |   ├── VSC # (DFT Ham.)
|   |   └── ...
|   ├──02-wfn
|   ├──03-wfnq
|   ├──05-band
|   ├──06-wfnq-nns
|   ├──07-aobasis
|   |   ├── ele1.ion # (LCAO basis)
|   |   ├── ele2.ion
|   |   └── ...
|   ├──11-epsilon
|   ├──11-epsilon-nns
|   ├──13-sigma
|   |   ├── eqp1.dat # (G0W0 corr.)
|   |   └── ...
|   ├──14-inteqp
|   ├──16-reconstruction
|   |   ├──aohamiltonian
|   |   |   ├── element.dat
|   |   |   ├── hamiltonians.h5
|   |   |   ├── info.json
|   |   |   ├── lat.dat
|   |   |   ├── orbital_types.dat
|   |   |   ├── overlaps.h5
|   |   |   ├── rlat.dat
|   |   └── └── site_positions.dat
|   ├──17-wfn_fi
|   ├──18-kernel
|   └──19-absorption
├── mat-2
|   └──  ...
└── ...
```

Related files on top of the folder:
- `QE, BGW, HPRO, SIESTA`: First-principle calculator
- `collect_tool.py`(see `-h`): 
    - deeph: `collect_tool.py deeph -flows FLOWS`
    - metalseek: `collect_tool.py metalseek -flows FLOWS `
    - st: `collect_tool.py st -flows FLOWS`
    - sub: `collect_tool.py sub -job JOB -hours HOURS -nodes NODES`
    - compact: `collect_tool.py compact -flows FLOWS (-folder FOLDER) (-unwanted UNWANTED)` (delete unwanted files for all flow and delete 02-wfn/wfn.h5 for all unifhished flow to save space)
    - restart: `collect_tool.py restart -flows FLOWS`
- `from_model/data.py` (**unit-test**): create for WFN, GW, BSE datatype
    - `from_model/wigner.py` (**unit-test**): create wigner cell for WFN
    - `from_model/interface.py` (**unit-test**): interface for `data.py`, including eqp, vloc, wfn, and AScvk classes



### 4. **ManyBodyData.h5** file
```
dataset.h5 (see data.py)
├── info/dict{}
├── mat-1/dict{}
├── mat-2/dict{}
├── mat-3/dict{}
```

Related files on top of the file:
- `collect_tool.py`(see `-h`): 
    - merge: `collect_tool.py merge -folder FOLDER -dataset_fname DATASET_FNAME` (merge all dataset h5 files into one)
- `from_model/data.py` (**unit-test**): load from h5 file
- `from_model/trainer.py`: train the model on the dataset
- `from_model/bsetrainer.py` (**unit-test**)
- `from_model/gwtrainer.py`
- `from_model/vaetrainer.py` (todo)
- `from_mode/wfnembedder.py` (**unit-test**)
  - create latent rep to manybodydata
  - create latent rep and save to manybodydata h5 file (suggested!)
  - parallel I/O

- models:
    - `from_model/transformer.py` (**unit-test**)
        - `from_model/basisassembly.py` (**unit-test**)
        - `from_model/posemb.py` (**unit-test**)
    - `from_model/e2vae.py` (**unit-test**)

### 5. **DeepH-E3** input folder

```
ml-train/test
├──graph_file (created by deep-preprocess.py)
├──ham1
|   ├── element.dat
|   ├── hamiltonians.h5
|   ├── info.json
|   ├── lat.dat
|   ├── orbital_types.dat
|   ├── overlaps.h5
|   ├── rlat.dat
|   └── site_positions.dat
├──ham2
|   └──  ...
└── ...
```
Related files on top of the folder:
see deeph3-train.py for more details.

### Benchmark

#### 1. data.py parallelization
| | interface.py  | data.py   | wall time |
|:----:|:------------:|:--------:|:-----------|
| **8 bands**          | -          | -      | <span style="color:red;">237s</span> (base line)    |
| | pool()     | -      | 232s      |
| | pool(4)    | -      | 218s      |
| | pool(8)    | -      | 217s      |
| | -          | pool() | <span style="color:green;">**30s**</span> (fast)      |
| **18 bands**| -          | pool() |  72s      |
| | pool(8)    | -      |     517s      |


## DDP Training for VAE

### Usage
Yaml files are used for inputs. Examples are included in the `config` folder.

1. Data preparation
Same as original code, but now supports yaml input directly.

```bash
deep-gwbse-gen-h5 -c data.yaml
```

2. Run DDP training for VAE. On interactive nodes:

```bash
srun -G 4 deep-gwbse-vae -c vae.yaml
```

To use bash script, a minimal submission script is:

```
#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH -t 30:00:00
#SBATCH -o dot.out

# ...environment preparation... (conda activate, etc.)

srun deep-gwbse-vae -c vae.yaml
```
Note that `-N` and `--gpus-per-node` should agree with input yaml, and `--ntasks-per-node` should be the same as tpus per node.

See [lightning docs](https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html) for more details.

During training, a directory `lightning_logs/version_X` will be created. It contains an event file for tensorboard monitoring and a checkpoint folders holding the checkpoint best on validation set. The checkpoint file `XXX.ckpt` contains more information than an ordinary model checkpoint. The model check point can be accessed as:
```python
ckpt = torch.load(ckpt_dir)
model_weights = ckpt["state_dict"]
```
The `model_weights` contains all the parameters, but all parameter names are prepended by `model.`. To load the checkpoint into a raw model (instead of a lightning module), the name should be modified:
```python
vae = EquivariantVAE(...)
for key in list(model_weights):
    model_weights[key.replace("model.", "")] = model_weights.pop(key)
vae.load_state_dict(model_weights)
```
Alternatively, one can also load the checkpoint into lightning module directly
```python
from deep_gwbse.from_model.vaetrainer_ddp import 
vae = EquivariantVAE(...)
vae_lightning = WFNVAETrainModule.load_from_checkpoint(ckpt_dir, model=model, ...other kwargs)
```
The module `vae_lightning` can be used as an ordinary `pytorch.nn.Module` such as evaluation `out = vae_lightnign(in)`.


3. Wavefunction Embedding & GW Training
```bash
deep-gwbse-embed -c embed.yaml
```
The resulting h5 dataset replace wfn data with latent dataa and can be used in GW Training as usual.

### Developer Notes

This section gives an introduction to all the new codes for Distributed Data Parallel (DDP) training. For more details, please see the documentation of the relavant codes and the documentation of [pytorch_lightning](https://lightning.ai/docs/pytorch/stable/).

The code is designed to be reusable at different levels. They are packed so that and there is no need to worry about parallelization explicitly. For a highly customized VAE, a minimal example DDP training script is as follows (run the code with `srun`):

```python
import pytorch_lightning as pL
import pytorch
from deep_gwbse.from_model.vaetrainer_ddp import WFNTrainModule

# step 1: define train module and collate_fn
class MyTrainModule(pL.LightningModule):
    pass
    # implements the logic of calculating loss functions (Ignore all machinery operations like zero_grad(), optimizer.step() / parallelization / loading to GPU, torch.no_grad() etc.), see the section below for an example.
    # Implement __init__, training_step, validation_step, test_step [optional].

    def training_step(self, batch, batch_idx):
        pass
        # batch comes from wfn_collate_fn output
    
    # ...

def wfn_collate_fn(batch):
    arr = batch[0]  # If original dataset is of shape (nk, nb, *trailing), then arr is an np.array of shape (N_wfn, *trailing), where N_wfn might be dynamic.
    pass

# step 2: initialize data
wfdata = ManyBodyData.from_existing_dataset("./wfndata.h5", lazy_load=True)  # this handles all parallelization for data

wfdata_lightning = WFNDataModule(
    base_dataset = wfdata,
    max_batch_size = 80,
    collate_fn=wfn_collate_fn,
    from_GW_data=True,
)

# step 3: initialize model and train module
vae = VAE(...)
vae_lightning = MyTrainModule(...)  # typically pass vae to __init__ of trainmodule

# step 4: initialize trainer
trainer = pL.Trainer(
    accelerator="gpu",
    num_nodes=1,
    devices=4,  # number of GPU per node
    max_epochs=1000,
    strategy="ddp",  # this handles all parallelization for training
    use_distributed_sampler=False, # MUST be False if use WFNDataModule!!!
)

# step 5: train
trainer.fit(vae_lightning, datamodule=wfdata_lightning)
# checkpoint will be automatic

# step 6: test
trainer.test(vae_lightning, datamodule=data_lightning)
```
A folder `lightning_logs/version_x` will be created. In the folder a tensorboard file will be created, and checkpoints will be saved to `lightning_logs/version_x/checkpoints`.


#### 1. Wavefunction Data Module

Defined in `vaetrainer_ddp.WFNDataModule`.

It supports complex datatypes.

This is the main module needed for VAE parallelization. It does two things: First, change the logic of the dataset (1 material as a datapoint -> A batch of wavefunctions as a datapoint, i.e., `(nk, nb, ...) -> several (N, ...)`). Second, use Lightning Data Module to prepare the dataloader for each process: The whole dataset will be split into several parts, and each process will only load its own part (This avoids CPU memory issues.). The Lightning Data Module can then be directly provided to lightning trainers.

In most cases `WFNDataModule` can be directly used. An example is:
```python
# initiate ManyBodyData defined in data.py
wfdata = ManyBodyData.from_existing_dataset("./wfndata.h5", data_slice=None, lazy_load=True)

# iniitate WFNDataModule, see explanation below.
data_lightning = WFNDataModule(
    base_dataset = wfdata,
    max_batch_size = 80,
    global_stage = "train",  # train / test
    train_stage_split = [0.8, 0.2, 0.0]  # train / val / test split during training global stage.
    collate_fn=wfn_collate_fn,
    from_GW_data=True,
)

# ... pass data_lightning to trainer
```
Requirements are:
1. `base_dataset`: wavefunctions should be stored in `[mat_id]/wfn` or `[mat_id]/src/wfn` (in the latter case set `from_GW_data` to True). The wavefunction data must be a numpy arrapy of shape `(nk, nb, ...)`, eigher real or complex dtype.
2. `max_batch_size`: The data module turns `(nk, nb, ...)` in to several `(N_wfn, ...)`. This arguments set the max `N_wfn`. Note that the actual `N_wfn` can be smaller than `max_batch_size`.
3. `global_stage`: If `train`, the dataset will be split into trian / val / test according to `train_stage_split`. Note that the test portion can be zero. If `test`, the whole dataset is used without splitting.
4. `collate_fn`: Must provides a callable. It should be called as `collate_fn(batch)`, where batch is a list of only **one** array of shape `(N_wfn, ...)` of the same dtype and trailing shape as the `wfndata`. Its output will be accepted by the lightning train module train / validation / test steps.
5. It is **highly reccomended** to use `lazy_load` in `ManyBodyData`. Otherwise each process will all try to read the whole dataset to CPU at once. `lazy_load` will suppress all loading. This is a new feature not in the original code. Loading will be handled by `WFNDataModule` where each process only load its own part of data.

#### 2. Wavefunction Train Module

Defined in `vaetrainer_ddp.WFNTrainModule`

This is a `LightningModule`, which is a subclass of `pytroch.nn.Module`. The current module is compatible with the real E2VAE and E3VAE. A custmozied train module might be needed for different VAEs.

Lightning Module is a highlevel wrap of all training logics. It automatically handles `.to("cuda")`, `.backward()`, trainer step, etc. Users only need to provide the logic of computing loss and the configuration of the optimizer. See pytorch lightning doc for details. A minimal example is:

```python
import pytroch_lightning as pL

class MyModule(pL.LightningModule):
    def __init__(self, model: torch.nn.Module,):
        # it is not necessary to pass the model as a whole.
        super().__init__()
        self.model = model
        self.mse = pytorch.nn.MSELoss()
    
    def training_step(self, batch, batch_idx):
        '''
        The signature MUST be: accept batch and batch_idx, return total_loss (a Tensor scaler).
       
        Args:
            batch: All the things produced by the collate function passed to the DataModule.
            batch_idx: index of the batch.
        
        Returns:
            loss (pytorch.Tensor): Scaler loss.
        '''
        # define whatever logics needed to comptue the loss
        x_in, label = batch
        predict = self.model(x_in)
        return self.mse(predict, label)

    def validation_step(self, batch, batch_idx):
        pass  # similar to training step

    def test_step(self, batch, batch_idx):
        pass  # similar to training step

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```


#### 3. I/O and Training
This is defined in `vaetrainer_ddp.run_vae`.

This mainly handles user I/O via yaml, initiates the DataModule, the model and the TrainModule, and then performs the training.

Developers might want to customize this part. A minimal example is as follows. `vaetrainer_ddp` also includes other logics such as resuming, load from checkpoint, etc.
```python
import pytorch_lightning as pL
# DDP train from scratch
trainer = pL.Trainer(
    accelerator="gpu",
    num_nodes=1,
    devices=4,  # number of GPU per node
    max_epochs=1000,
    strategy="ddp",
    use_distributed_sampler=False, # MUST be False if use WFNDataModule!!!
)
trainer.fit(vae_lightning, datamodule=data_lightning)  # data_lightning: instance of WFNDataModule, vae_lightning: instance of LightningModule.
```

Note: **If using `WFNDataModule`, `use_distributed_sampler` MUST be set to False in the trainer!!!** This is because `WFNDataModule` already implements a distribution logic slightly different from the raw pytorch DDP (here we let each process only load a fixed part of the whole data), and setting `use_distributed_sampler` while using `WFNDataModule` will split the data twice, resulting in undefined behavior.

If developers wish to directly use the currently implemeted logic, `run_vae` can be directly called. See the comments in `vaetrainer_ddp.run_vae` for details. To use `run_vae`, please only customize the `model` part of the config file.


#### 4. Other codes

**Data Module**
Defined in `trainer.TrainDataModule`. Used by the `WFNDataModule`. No need to touch it.


This is a Lightning Data module served as an abstract class. Lightning data modules produce train / validation / test / prediction dataloaders and can be directly passsed to a lightning trainer. Two features implemented in `TrainDataModule`:

1. Split the dataset across different process (for DDP): `TrainDataModule._split_process`. This is a function modified from torch `DDPSampler`. It takes an integer (total dataset length) and returns a list of indices for the current process.

2. Train/Val/Test split: `TrainDataModule.setup` and `TrainDataModule.train/val/test/predict_dataloader`. It splits the whole dataset into train / val / test / predict based on global_stage: train / test/ predict and prepares the dataloaders.

**SubBatchWFNData**
Defined in `vaetrainer_ddp.SubBatchWFNData`. Used by the `WFNDataModule`. No need to touch it.

It is a pytorch `Dataset` responsible for the logic change `(nk, nb, ...) -> (N_wfn, ...)`.

**Miscellaneous**
1. A `lazy_load` feature is added to `ManybodyData`, it initializes mat_id but skip all loading. A datapoint will be loaded from h5 only when required via indexing.
2. `vaetrainer.wfn_collate_fn` and `vaetrainer.wfn_collate_fn_3D`: a `batched_data` argument is added to be compatible with `WFNDataModule`. The defualt is compatible with old codes.
3. Move `torch.backends.cudnn.enabled = False` to the main part of `vaetrainer.py`. This is for 3D convolution only.
4. A seperate `kl_loss_mean` function is added to `e2vae` for the lightning module to report both mse and KL divergence.