from .utils import decode_outputs

class Anonymizer:
    def __init__(self, config, classifier):
        self.config = config
        self.classifier = classifier

    def get_identifiable_tokens(self, text_input):
        predictions = decode_outputs(
            self.classifier(text_input), model_type=self.config.model_type
        )
        return predictions

    def anonymize(self, text_input):
        # TODO: Implement replacement
        tokens = self.get_identifiable_tokens(text_input)
        return text_input
