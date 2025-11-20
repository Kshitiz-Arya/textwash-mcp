import re
from copy import deepcopy
from .utils import decode_outputs

class Anonymizer:
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier
        with open(self.config.path_to_months_file, "r") as f:
            self.months = [m.strip() for m in f.readlines()]
        with open(self.config.path_to_written_numbers_file, "r") as f:
            self.written_numbers = [w.strip() for w in f.readlines()]
        self.valid_surrounding_chars = [".", ",", ";", "!", ":", "\n", "’", "‘", "'", '"', "?", "-"]

    def analyze(self, text_input):
        predictions = decode_outputs(self.classifier(text_input), model_type=self.config.model_type)
        return [p for p in predictions if p["entity"] != "NONE" and len(p["word"]) > 1 and p["word"].isalnum()]

    def replace_heuristics(self, text):
        # Numeric Heuristics
        all_numeric = list(set(re.findall("[0-9]+", text)))
        numeric_map = {k: "NUMERIC_{}".format(v + 1) for v, k in enumerate(all_numeric)}
        for k, v in sorted(numeric_map.items(), key=lambda x: int(x[0]), reverse=True):
            text = re.sub(f"(?<![0-9]){k}(?![0-9])", f" {v}", text)
            
        # Pronoun Heuristics
        pronoun_map = {"he": "PRONOUN", "she": "PRONOUN", "mr": "MR/MS", "dr": "TITLE"} # Simplified list for brevity
        for k, v in pronoun_map.items():
            text = re.sub(r'\b' + re.escape(k) + r'\b', v, text, flags=re.IGNORECASE)
        return text

    def anonymize(self, input_seq, selected_entities=None, strategy="standard", return_mapping=False):
        raw_entities = self.analyze(input_seq)
        entities = {p["word"]: p["entity"] for p in raw_entities}
        
        if selected_entities:
            entities = {k: v for k, v in entities.items() if v in selected_entities}

        counts = {}
        mapping = {}
        for word, etype in entities.items():
            if strategy == "redact":
                mapping[word] = "[REDACTED]"
            else:
                if etype not in counts: counts[etype] = 1
                mapping[word] = f"{etype}_{counts[etype]}"
                counts[etype] += 1
            
        for phrase, replacement in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
             input_seq = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, input_seq)
        
        if strategy != "redact":
            input_seq = self.replace_heuristics(input_seq)

        if return_mapping:
            return input_seq, mapping
        return input_seq
