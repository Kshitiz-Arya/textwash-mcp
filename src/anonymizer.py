import re
from copy import deepcopy
from .utils import decode_outputs

class Anonymizer:
    """
    Main class for detecting and anonymizing PII (Personally Identifiable Information) in text.
    Uses ML models for entity detection and rule-based heuristics for additional coverage.
    """
    
    def __init__(self, config, classifier):
        """Initialize anonymizer with configuration and ML classifier."""
        self.config = config
        self.classifier = classifier  # ML model for named entity recognition
        
        # Load reference data for temporal and numeric entity detection
        with open(self.config.path_to_months_file, "r") as f:
            self.months = [m.strip() for m in f.readlines()]
        with open(self.config.path_to_written_numbers_file, "r") as f:
            self.written_numbers = [w.strip() for w in f.readlines()]
            
        # Characters that can surround PII entities (currently unused but available for future features)
        self.valid_surrounding_chars = [".", ",", ";", "!", ":", "\n", "'", "'", "'", '"', "?", "-"]

    def analyze(self, text_input):
        """Detect PII entities in text using the ML classifier.
        
        Returns list of entities with word, position, and type information.
        Filters out non-entities, single characters, and non-alphanumeric tokens.
        """
        # Get predictions from ML model and fix tokenization issues
        predictions = decode_outputs(self.classifier(text_input), model_type=self.config.model_type)
        
        # Filter to keep only valid PII entities
        return [p for p in predictions if p["entity"] != "NONE" and len(p["word"]) > 1 and p["word"].isalnum()]

    def replace_heuristics(self, text):
        """Apply rule-based anonymization for patterns the ML model might miss.
        
        Handles numeric sequences and common pronouns/titles.
        """
        # Find and replace numeric sequences with numbered placeholders
        all_numeric = list(set(re.findall("[0-9]+", text)))  # Get unique number sequences
        numeric_map = {k: "NUMERIC_{}".format(v + 1) for v, k in enumerate(all_numeric)}
        
        # Replace numbers (largest first to avoid partial matches)
        for k, v in sorted(numeric_map.items(), key=lambda x: int(x[0]), reverse=True):
            # Use negative lookbehind/lookahead to match standalone numbers only
            text = re.sub(f"(?<![0-9]){k}(?![0-9])", f" {v}", text)
            
        # Replace common pronouns and titles that might indicate gender or professional status
        pronoun_map = {"he": "PRONOUN", "she": "PRONOUN", "mr": "MR/MS", "dr": "TITLE"} # Simplified list for brevity
        for k, v in pronoun_map.items():
            # Match whole words only, case-insensitive
            text = re.sub(r'\b' + re.escape(k) + r'\b', v, text, flags=re.IGNORECASE)
        return text

    def anonymize(self, input_seq, selected_entities=None, strategy="standard", return_mapping=False):
        """Main anonymization method.
        
        Args:
            input_seq: Text to anonymize
            selected_entities: List of entity types to anonymize (None = all)
            strategy: 'standard' (numbered placeholders) or 'redact' ([REDACTED])
            return_mapping: Whether to return the replacement mapping
        
        Returns:
            Anonymized text, optionally with mapping dictionary
        """
        # Find all PII entities in the text
        raw_entities = self.analyze(input_seq)
        entities = {p["word"]: p["entity"] for p in raw_entities}
        
        # Filter to only selected entity types if specified
        if selected_entities:
            entities = {k: v for k, v in entities.items() if v in selected_entities}

        # Build replacement mapping
        counts = {}  # Track count of each entity type for numbering
        mapping = {}
        for word, etype in entities.items():
            if strategy == "redact":
                mapping[word] = "[REDACTED]"
            else:
                # Create numbered placeholders (PERSON_1, PERSON_2, etc.)
                if etype not in counts: counts[etype] = 1
                mapping[word] = f"{etype}_{counts[etype]}"
                counts[etype] += 1
            
        # Apply replacements (longest phrases first to avoid partial matches)
        for phrase, replacement in sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True):
             input_seq = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, input_seq)
        
        # Apply heuristic rules for additional coverage (skip for redaction strategy)
        if strategy != "redact":
            input_seq = self.replace_heuristics(input_seq)

        if return_mapping:
            return input_seq, mapping
        return input_seq
