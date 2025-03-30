import re
from zipfile import ZipFile


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

        for i, measure in enumerate(self.measures):
            measure_notes = []
            measure_content = self.extract_measure_content(measure)

            for symbol in measure_content:
                if symbol.startswith("<beam"):
                    beam_notes = re.findall(r"<note.[\s\S]*?note>|<note.*?/>", symbol)
                    measure_notes.append("".join(self.parse_note(n) for n in beam_notes))
                elif symbol.startswith("<note"):
                    measure_notes.append(self.parse_note(symbol))
                elif symbol.startswith("<rest"):
                    duration = self.duration_mapping[
                        re.search(r'dur="([^"]*)"', symbol).group(1)
                    ]
                    measure_notes.append(f"z{duration}")
                elif symbol.startswith("<multiRest"):
                    duration = re.search(r'num="([^"]*)"', symbol).group(1)
                    measure_notes.append(f"Z{duration}")

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

    @staticmethod
    def convert_zip(zip_path, number_of_files=-1):
        """
        Converts all MEI files in a ZIP archive to ABC notation.

        :param zip_path: Path to the ZIP archive.
        :return: List of MEIConverter instances.
        """
        converters = []
        with ZipFile(zip_path, "r") as myzip:
            for myfile in myzip.infolist()[:number_of_files]:
                if myfile.filename.endswith(".mei"):
                    with myzip.open(myfile) as f:
                        file_content = f.read().decode("utf-8")
                        converters.append(MEIConverter(content=file_content))

        return converters
