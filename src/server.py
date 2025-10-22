import torch
from mcp.server.fastmcp import FastMCP
from .config import Config
from .anonymizer import Anonymizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

mcp = FastMCP("Textwash Anonymizer")

def load_model():
    config = Config("en")
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(config.path_to_model)
    model = AutoModelForTokenClassification.from_pretrained(config.path_to_model)
    classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    return Anonymizer(config, classifier)

anonymizer = load_model()

@mcp.tool()
def anonymize(text: str) -> str:
    return anonymizer.anonymize(text)

if __name__ == "__main__":
    mcp.run()
