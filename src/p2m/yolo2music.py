from music21 import converter, braille, midi, instrument, tempo
from p2m.model import predict
import re

def abc_to_braille(abc_file):
    # Load ABC file
    abc_score = converter.parse(abc_file, format='abc')
    braille_rep = braille.translate.objectToBraille(abc_score)
    return braille_rep

def abc_to_midi_and_play(abc_file, output_file=None, play=False, instrument_class=None, tempo_bpm=None):
    # Load ABC file
    abc_score = converter.parse(abc_file, format='abc')

    if instrument_class:
        # Create a Stream directly and apply instrument to the elements in the stream
        abc_score.parts[0].insert(0, instrument_class())

    # Set the tempo if provided
    if tempo_bpm:
        tempo_marking = tempo.MetronomeMark(number=tempo_bpm)
        abc_score.insert(0, tempo_marking)

    if output_file:
        # Write the score to a MIDI file
        abc_score.write('midi', fp=output_file)
        print(f"MIDI file saved to {output_file}")
    
    if play:
        # Play the MIDI file
        player = midi.realtime.StreamPlayer(abc_score)
        player.play()
    
def abc_to_image(abc_file):
    # Load ABC file
    abc_score = converter.parse(abc_file, format='abc')

    # Show the musical score
    abc_score.show('musicxml.png')

def write_abc(notes: list):
    """
    Writes a list of notes to an ABC file.
    :param notes: List of notes to write to the ABC file.
    """
    abc = ' '.join(notes)
    return abc

def yolo2abc(result):
    """
    Sorts YOLO predictions by x-coordinate and returns class names.

    :param result : YOLO prediction object
    :return: List of detected notes in sorted x order
    """
    gammes = {
        "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
        "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
        "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
    }
    # Convert tensors to lists
    cls_list = result.boxes.cls.tolist()
    data_list = result.boxes.data.tolist()
    class_names = result.names

    # Combine class IDs and bounding box info
    detections = list(zip(cls_list, data_list))

    # Sort detections by x-coordinate (first value in `data_list`)
    sorted_detections = sorted(detections, key=lambda x: x[1][0])

    # Get class names in sorted order
    sorted_notes = [class_names[int(d[0])] for d in sorted_detections]

    if sorted_notes[0] in ['C1', 'C2', 'C3', 'C4', 'G2', 'F3', 'F4']:
        sorted_notes[0] = f'K: clef={sorted_notes[0]}\n'
    if sorted_notes[1] in gammes.values():
        sorted_notes[1] = f'K: {sorted_notes[1]}\n' 
    if bool(re.match(r'^\d+/\d+$', sorted_notes[2])):
        sorted_notes[2] = f'M: {sorted_notes[2]}\n'

    abc = write_abc(sorted_notes)

    return abc

if __name__ == "__main__":
    model_path = "models/chopin.pt"
    image_path = "data/YOLO_small_1000/test/images/000101766-1_4_1.png"
    result = predict(model_path, image_path)[0]
    abc = yolo2abc(result)

    print(abc)


    # mei = MEIConverter('resources/dataset/batch_1/match/labels/000100536-1_1_1.mei')
    # abc = mei.mei_to_abc()

    abc_to_midi_and_play(abc, instrument_class = instrument.PanFlute, play=True)
    # abc_to_braille(abc)
    abc_to_image(abc)
    # print(instrument.__dict__.keys()) 
