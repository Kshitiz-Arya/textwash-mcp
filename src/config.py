import os
from pathlib import Path

class Config:
    def __init__(self, language: str):
        if language not in ["nl", "en"]:
            raise ValueError(f"Invalid language '{language}' specified")
            
        base_dir = Path(__file__).parent.parent.resolve()
        data_dir = base_dir / "data"

        self.path_to_months_file = str(data_dir / "months.txt")
        self.path_to_written_numbers_file = str(data_dir / "written_numbers.txt")
        self.path_to_model = str(data_dir / language)
        self.model_type = "bert" if language == "nl" else "roberta"
        
        if not os.path.exists(self.path_to_model):
            raise FileNotFoundError(f"Model not found at {self.path_to_model}")
