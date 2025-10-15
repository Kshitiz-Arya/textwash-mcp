import re
from .utils import decode_outputs

class Anonymizer:
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier
        with open(self.config.path_to_months_file, "r") as f:
            self.months = [m.strip() for m in f.readlines()]
        with open(self.config.path_to_written_numbers_file, "r") as f:
            self.written_numbers = [w.strip() for w in f.readlines()]

    def anonymize(self, text_input):
        predictions = decode_outputs(self.classifier(text_input), model_type=self.config.model_type)
        entities = {p["word"]: p["entity"] for p in predictions if p["entity"] != "NONE"}
        
        # Simple replacement
        for word, ent in entities.items():
            text_input = text_input.replace(word, f"<{ent}>")
            
        return text_input
