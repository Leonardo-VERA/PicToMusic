import re

def yolo_to_abc(results):
    """
    Converts multiple YOLO predictions into ABC notation format.

    :param results: List of YOLO prediction objects, one for each staff line
    :return: ABC notation string
    """
    gammes = {
        "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
        "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
        "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
    }

    if not results:
        return "No notes detected in the image."

    # Initialize ABC header
    abc_content = [
        "X:1",          # Reference number
        "T:",           # Title (blank for now)
        "M:4/4",        # Default time signature
        "L:1/8",        # Default note length
        "Q:1/4=120",    # Default tempo
        "K:C",          # Default key signature
    ]

    for i, result in enumerate(results):
        cls_list = result.boxes.cls.tolist()
        data_list = result.boxes.data.tolist()
        class_names = result.names

        if not cls_list or not data_list:
            continue

        detections = list(zip(cls_list, data_list))
        sorted_detections = sorted(detections, key=lambda x: x[1][0])
        sorted_notes = [class_names[int(d[0])] for d in sorted_detections]

        if i == 0:
            # Handle clef
            if len(sorted_notes) > 0 and sorted_notes[0] in ['C1', 'C2', 'C3', 'C4', 'G2', 'F3', 'F4']:
                clef = sorted_notes[0]
                abc_content.append(f"%%clef {clef}")

            # Handle key signature
            if len(sorted_notes) > 1 and sorted_notes[1] in gammes.values():
                key = sorted_notes[1]
                abc_content[5] = f"K:{key}"

            # Handle time signature
            if len(sorted_notes) > 2 and bool(re.match(r'^\d+/\d+$', sorted_notes[2])):
                time_sig = sorted_notes[2]
                abc_content[2] = f"M:{time_sig}"

        # Process notes for this line
        notes = sorted_notes[3:] if len(sorted_notes) > 3 else []
        if notes:
            measure = []
            current_measure = []
            for note in notes:
                current_measure.append(note)
                if len(current_measure) >= 8:  # 8 eighth notes = 4 beats
                    measure.append(' '.join(current_measure))
                    current_measure = []
            
            if current_measure:
                measure.append(' '.join(current_measure))
            
            abc_content.extend(measure)

    abc_content.append('|]')

    return '\n'.join(abc_content)