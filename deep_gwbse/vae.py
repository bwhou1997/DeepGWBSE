import argparse
import yaml
from .from_model.vaetrainer_ddp import main as vae_main
from pytorch_lightning.utilities import rank_zero_info


def real_vae():
    rank_zero_info("E2VAE Trainer Complex.")

    # read input
    parser = argparse.ArgumentParser(description='Gauge equivariant version of Wavefunction E2VAE.')
    parser.add_help = True
    parser.add_argument('-c', '--config', type=str, default='./vae.yaml', help='input config yaml file.')
    config_dir = parser.parse_args().config
    rank_zero_info(f"Read input config from: {config_dir}")
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)

    vae_main(config)

