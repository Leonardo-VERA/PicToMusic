import re
from zipfile import ZipFile
import os


class MEIConverter:
    """
    A class to convert MEI files to ABC notation.
    """

    octaves = {0: ",,,,,", 1: ",,,,", 2: ",,,", 3: ",,", 4: ",", 5: "", 6: "'", 7: "''"}
    duration_mapping = {'128': "///", '64': "//", '32': "/", '16': 1, '8': 2, '4': 4, '2': 8, '1': 16, 'breve': 32, 'long':64}
    accid_map = {"s": "^", "f": "_", "n": "="}
    gammes = {
        "0": "C", "1s": "G", "2s": "D", "3s": "A", "4s": "E", "5s": "B",
        "6s": "F#", "7s": "C#", "1f": "F", "2f": "Bb", "3f": "Eb",
        "4f": "Ab", "5f": "Db", "6f": "Gb", "7f": "Cb"
    }

    def __init__(self, file_name: str = None, content: str = None):
        """
        Initializes the MEIConverter with file content.

        :param file_name: Path to the MEI file (optional).
        :param content: Raw MEI file content (optional).
        """
        self.file_content = content if content else self.read_file(file_name)
        self.measures = []
        self.measures_content = {}
        self.abc_content = ""
        self.score_def = self.find_score_def()
        self.find_measures()

    def read_file(self, file_name):
        """
        Reads the file content from the given file name.
        """
        with open(file_name, "r", encoding="utf-8") as f:
            return f.read()

    def find_measures(self):
        """
        Finds and extracts all measures from the MEI content.
        """
        self.measures = re.findall(r"<measure[\s\S]*?measure>", self.file_content)

    def extract_measure_content(self, measure):
        """
        Extracts notes, rests, and beams from a given measure.

        :param measure: MEI measure content.
        :return: List of extracted musical elements.
        """
        return re.findall(
            r"<beam[\s\S]*?beam>|<note.*?/>|<note.[\s\S]*?note>|<rest.*?/>|<multiRest.*?/>",
            measure,
        )

    def parse_note(self, note):
        """
        Parses a note from MEI format to ABC notation.

        :param note: Raw MEI note element.
        :return: ABC notation string.
        """
        value = re.search(r'pname="([^"]*)"', note).group(1)
        octave = self.octaves[int(re.search(r'oct="([^"]*)"', note).group(1))]
        duration = self.duration_mapping[re.search(r'dur="([^"]*)"', note).group(1)]

        if 'dots="' in note:
            if duration == '/':
                duration = '3/4'
            elif duration == '//':
                duration = '3/8'
            else:
                duration = f"{int(duration * 1.5)}"



        accid_match = re.search(r'accid="([^"]*)"', note)
        if accid_match:
            value = f"{self.accid_map.get(accid_match.group(1), '')}{value}"

        return f"{value}{octave}{duration}"

    def get_measures_labels(self):
        """
        Extracts labels for each measure and stores them.
        """
        self.measures_content = {}
        self.notes_labels = []
        self.pause_labels = []

        for i, measure in enumerate(self.measures):
            measure_notes = []
            measure_content = self.extract_measure_content(measure)

            for symbol in measure_content:
                if symbol.startswith("<beam"):
                    beam_notes = re.findall(r"<note.[\s\S]*?note>|<note.*?/>", symbol)
                    beam_notes_labels = [self.parse_note(n) for n in beam_notes]
                    measure_notes.append("".join(n for n in beam_notes_labels))
                    self.notes_labels += beam_notes_labels
                elif symbol.startswith("<note"):
                    note_label = self.parse_note(symbol)
                    measure_notes.append(note_label)
                    self.notes_labels.append(note_label)
                elif symbol.startswith("<rest"):
                    duration = self.duration_mapping[
                        re.search(r'dur="([^"]*)"', symbol).group(1)
                    ]
                    measure_notes.append(f"z{duration}")
                    self.pause_labels.append(f"z{duration}")
                elif symbol.startswith("<multiRest"):
                    duration = re.search(r'num="([^"]*)"', symbol).group(1)
                    measure_notes.append(f"Z{duration}")
                    self.pause_labels.append(f"Z{duration}")

            self.measures_content[i] = measure_notes

    def find_score_def(self):
        """
        Extracts score definitions (key, meter, clef) from the MEI file.

        :return: A dictionary with key, meter, and clef information.
        """
        score_def = re.findall(r"<scoreDef[\s\S]*?scoreDef>", self.file_content)
        if not score_def:
            return {"key": "", "meter_count": "", "meter_unit": "", "clef": ""}

        key = re.findall(r'key.sig="([^"]*)"', score_def[0])
        meter_count = re.findall(r'meter.count="([^"]*)"', score_def[0])
        meter_unit = re.findall(r'meter.unit="([^"]*)"', score_def[0])
        staff_defs = re.findall(r"<staffDef.*?/>", score_def[0])

        key = self.gammes[key[0]] if key else ""
        meter_count = int(meter_count[0]) if meter_count else ""
        meter_unit = int(meter_unit[0]) if meter_unit else ""

        clef_shape = ""
        clef_line = ""
        if staff_defs:
            clef_shape_match = re.search(r'clef.shape="([^"]*)"', staff_defs[0])
            clef_line_match = re.search(r'clef.line="([^"]*)"', staff_defs[0])
            if clef_shape_match:
                clef_shape = clef_shape_match.group(1)
            if clef_line_match:
                clef_line = clef_line_match.group(1)

        return {"key": key, "meter_count": meter_count, "meter_unit": meter_unit, "clef": clef_shape + clef_line}

    def mei_to_abc(self):
        """
        Converts the MEI file content to ABC notation.

        :return: ABC notation string.
        """
        abc_content = "X:1\n"
        abc_content += (
            f'M:{self.score_def["meter_count"]}/{self.score_def["meter_unit"]}\n'
            if self.score_def["meter_count"]
            else "M:C\n"
        )
        abc_content += f'K:{self.score_def["key"]}\n' if self.score_def["key"] else ""
        abc_content += f"L:1/16\n"
        abc_content += f'K: clef={self.score_def["clef"]}\n'

        self.get_measures_labels()

        for measure_notes in self.measures_content.values():
            abc_content += " ".join(measure_notes) + " |"
        abc_content += "]"
        self.abc_content = abc_content
        return abc_content
    
    def treble_clef_transposition(self):
        
        notes = []
        for note in self.notes_labels:
            notes.append(self.convert_note_to_treble(self.score_def["clef"], note))
       
        return notes
    
    
    @staticmethod    
    def convert_note_to_treble(clef, note):
        """
        Convert a note from a given clef to its equivalent in the treble clef.
        This also ensures to handle octaves correctly and strips until the first numerical character.
        """

        clef_to_treble = {
        "C1": {  # C on the 1st line
            'a,': 'c',  'b,': 'd',  'c': 'e',  'd': 'f',  'e': 'g',  
            'f': 'a',  'g': 'b',  'a': "c'",  'b': "d'",
        },
        "C2": {  # C on the 2nd line
            'f,': 'c',  'g,': 'd',  'a,': 'e',  'b,': 'f',  'c': 'g',  
            'd': 'a',  'e': 'b',  'f': "c'",  'g': "d'",  'a': "e'",  'b': "f'",
        },
        "C3": {  # C on the 3rd line
            'd,': 'c',  'e,': 'd',  'f,': 'e',  'g,': 'f',  'a,': 'g',  
            'b,': 'a',  'c': 'b',  'd': "c'",  'e': "d'",  'f': "e'",  'g': "f'",  'a': "g'",  'b': "a'",
        },
        "C4": {  # C on the 4th line
            'b,,': 'c',  'c,': 'd',  'd,': 'e',  'e,': 'f',  'f,': 'g',  
            'g,': 'a',  'a,': 'b',  'b,': "c'",  'c': "d'",  'd': "e'",  'e': "f'",  'f': "g'",  'g': "a'",  'a': "b'",  'b': "c''",
        },
        "F4": {  # F on the 4th line
            'e,,': 'c',  'f,,': 'd',  'g,,': 'e',  'a,,': 'f',  'b,,': 'g',  'c,': 'a',  
            'd,': 'b', 'e,': "c'",  'f,': "d'",  'g,': "e'",  'a,': "f'",  'b,': "g'",  'c': "a'",  'd': "b'",  'e': "c''",  'f': "d''",  'g': "e''",  'a': "f''",  'b': "g''"
        },
        "F3": {  # F on the 3rd line
            'e,,':'a,',  'f,,': 'b,',  'g,,': 'c',  'a,,': 'd',  'b,,': 'e',  'c,': 'f',  
            'd,': 'g', 'e,': 'a',  'f,': 'b',  'g,': "c'",  'a,': "d'",  'b,': "e'",  'c': "f'",  'd': "g'",  'e': "a'",  'f': "b'",  'g': "c''",  'a': "d''",  'b': "e''"
        },
    }

        if clef not in clef_to_treble:
            return note  # If clef is not in mapping, return the note as is.

        # Get the mapping for the current clef
        clef_mapping = clef_to_treble[clef]

        # Sort keys of the clef mapping by length in descending order
        sorted_keys = sorted(clef_mapping.keys(), key=lambda x: len(x), reverse=True)

        # Create a regex pattern that matches the note names in the order of longest to shortest
        pattern = '|'.join(re.escape(key) for key in sorted_keys)

        # Function to replace the matched note with its corresponding value
        def replace_note(match):
            matched_note = match.group(0)
            return clef_mapping.get(matched_note, matched_note)  # Return mapped value or original note

        # Use re.sub to replace the matched notes with corresponding values
        return re.sub(pattern, replace_note, note)
        

    @staticmethod
    def convert_zip(zip_path, number_of_files=-1):
        """
        Converts all MEI files in a ZIP archive to ABC notation.

        :param zip_path: Path to the ZIP archive.
        :return: List of MEIConverter instances.
        """
        converters = []
        with ZipFile(zip_path, "r") as myzip:
            # Get a list of all MEI files in the "labels/" folder
            mei_files = [f for f in myzip.namelist() if f.startswith("labels/") and f.endswith(".mei")]
            
            # Limit the number of files to process
            for mei_file in mei_files[:number_of_files]:
                with myzip.open(mei_file) as f:
                    file_content = f.read().decode("utf-8")
                    converters.append(MEIConverter(content=file_content))

        return converters


# Example usage:
# converter = MEIConverter('../../pic2music/000051778-1_1_1.mei')
# print(converter.mei_to_abc())

# To convert a ZIP file
# converters = MEIConverter.convert_zip("resources/dataset/dataset.zip", number_of_files=2000)

# from tqdm import tqdm
# # Access ABC notation of each MEI file
# for conv in tqdm(converters):
#     if conv.score_def["clef"] not in ["G2", "C1", "F4", "C2", "C3", "C4", "G1", "F3"]:
#         print(conv.score_def["clef"])
