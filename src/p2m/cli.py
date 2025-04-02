import click
import yaml
from p2m import __version__

@click.group(name='p2m', help='A command line tool to convert images to music')
@click.version_option(__version__)
def p2m():
    pass

@p2m.group(name='model', help='Set of commands to launch training, prediction or evaluation models')
def model():
    pass

@model.command(name='train', help='Train a YOLO model on a custom dataset')
@click.option('--data-path', '-d', required=True, help='Path to dataset configuration file (e.g. data/dataset.yaml)')
@click.option('--model-path', '-m', default='yolo11n.pt', help='Path to initial model weights')
@click.option('--config-path', '-c', default='configs/training_config.yaml', help='Path to training configuration YAML file')
def train(data_path: str, model_path: str, config_path: str):
    """Execute the YOLO model training workflow with specified parameters."""
    import p2m.model
    
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    p2m.model.train(
        data_path=data_path,
        model_path=model_path,
        **training_config
    )