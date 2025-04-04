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

@p2m.command(name='play-yolo', help='Play a MIDI file from YOLO predictions')
@click.option('--image-path', '-i', required=True, help='Path to the image file for YOLO predictions')
@click.option('--model-path', '-m', required=True, help='Path to the trained YOLO model')
@click.option('--instrument_class', '-inst', default='Piano', help='Instrument to use for the MIDI output (e.g., PanFlute, Piano, Violin)')
@click.option('--tempo', '-t', default=120, type=int, help='Tempo for the MIDI output in beats per minute (default: 120)')
def play_midi_from_yolo(image_path: str, model_path: str, instrument_class: str, tempo: int):
    """Generate MIDI from YOLO predictions and play it."""
    import p2m.model
    import p2m.yolo2music
    from music21 import instrument

    try:
        instrument_class = getattr(instrument, instrument_class)
    except AttributeError:
        print(f"Invalid instrument name '{instrument_class}', falling back to 'Piano'.")
        instrument_class = instrument.Piano  # Fallback to Piano if not found

    # Run YOLO prediction
    result = p2m.model.predict(model_path, image_path)[0]
    
    # Convert the predictions to ABC notation
    abc = p2m.yolo2music.yolo2abc(result)

    # Play and save the MIDI file
    p2m.yolo2music.abc_to_midi_and_play(abc, play=True, instrument_class=instrument_class, tempo_bpm=tempo)