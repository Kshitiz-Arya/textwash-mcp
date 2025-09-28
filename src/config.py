import os
from pathlib import Path

class Config:
    def __init__(self, language: str):
        self.language = language
        # Base directory resolution
        base_dir = Path(__file__).parent.parent.resolve()
        data_dir = base_dir / "data"

        self.path_to_months_file = str(data_dir / "months.txt")
        self.path_to_written_numbers_file = str(data_dir / "written_numbers.txt")
        self.path_to_model = str(data_dir / language)
        self.model_type = "bert" if language == "nl" else "roberta"
