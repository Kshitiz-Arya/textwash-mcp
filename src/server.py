import torch
from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from .config import Config
from .anonymizer import Anonymizer

mcp = FastMCP("Textwash Anonymizer")
_MODEL_CACHE = {}

def get_anonymizer(language: str):
    if language in _MODEL_CACHE:
        return _MODEL_CACHE[language]

    print(f"Loading model for {language}...")
    config = Config(language=language)
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(config.path_to_model)
    model = AutoModelForTokenClassification.from_pretrained(config.path_to_model)
    classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    
    anonymizer = Anonymizer(config, classifier)
    _MODEL_CACHE[language] = anonymizer
    return anonymizer

@mcp.tool()
def anonymize_text(text: str, language: str = "en") -> str:
    anonymizer = get_anonymizer(language)
    return anonymizer.anonymize(text)

if __name__ == "__main__":
    mcp.run()
