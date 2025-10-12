from .utils import decode_outputs

class Anonymizer:
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier
        
        with open(self.config.path_to_months_file, "r") as f:
            self.months = [m.strip() for m in f.readlines()]
        with open(self.config.path_to_written_numbers_file, "r") as f:
            self.written_numbers = [w.strip() for w in f.readlines()]

    def get_identifiable_tokens(self, text_input):
        predictions = decode_outputs(
            self.classifier(text_input), model_type=self.config.model_type
        )
        return predictions

    def anonymize(self, text_input):
        return text_input
