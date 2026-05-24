import argparse
import yaml
from deep_gwbse.from_model.data import ManyBodyData
from deep_gwbse.from_model.wfnembedder import LightningEmbedder_realVAE, ManyBodyData_WFN_Embedder_pretrained


def gen_h5_data():
    print("Generate h5 dataset.", flush=True)
    # read input
    parser = argparse.ArgumentParser(description='Generate h5 dataset from flows.')
    parser.add_help = True
    parser.add_argument('-c', '--config', type=str, default='./data.yaml', help='input config yaml file.')
    config_dir = parser.parse_args().config
    print("Read input config from:", config_dir)
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)
    
    # prepare dataset
    dataset_type = config["system"]["dataset_type"].upper()
    dataset_params = config["system"]
    dataset_params["dataset_type"] = dataset_type  # change to upper case
    dataset_params["load_dataset"] = False
    dataset_params["onlySave"] = True

    # if dataset_type 
    if dataset_type not in ["WFN", "GW", "BSE"]:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types: WFN, GW, BSE.")
    dataset_params |= config["WFN"]
    if dataset_type == "GW":
        dataset_params |= config["GW"]
    if dataset_type == "BSE":
        dataset_params |= config["BSE"]
    
    if dataset_type == "GW" or dataset_type == "BSE":
        dataset_params["from_dft"] = True

    print("Start.", flush=True)    
    ManyBodyData(**dataset_params)


def embed():
    '''wfn to latent'''
    parser = argparse.ArgumentParser(description='Embed wavefunction to latent space.')
    parser.add_help = True
    parser.add_argument('-c', '--config', type=str, default='./embed.yaml', help='input config yaml file.')
    embed_config_dir = parser.parse_args().config
    with open(embed_config_dir, 'r') as f:
        embed_config = yaml.safe_load(f)

    manybodydata = ManyBodyData.from_existing_dataset(embed_config["orig_dataset_dir"], lazy_load=embed_config["lazy_load"])
    embedder = ManyBodyData_WFN_Embedder_pretrained(None, LightningEmbedder_realVAE,
                                                    config_dir=embed_config["model_config_dir"],
                                                    ckpt_dir=embed_config["ckpt_dir"],
                                                    device=embed_config["device"],
                                                    )

    embedder.create_latent_for_ManyBodyData_h5(manybodydata, dataset_dir=embed_config["save_dataset_dir"],
                                               dataset_fname=embed_config["save_dataset_fname"])
    print("DONE")
    