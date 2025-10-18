from mcp.server.fastmcp import FastMCP
from .config import Config
from .anonymizer import Anonymizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

mcp = FastMCP("Textwash Anonymizer")

# Global instance
config = Config("en")
tokenizer = AutoTokenizer.from_pretrained(config.path_to_model)
model = AutoModelForTokenClassification.from_pretrained(config.path_to_model)
classifier = pipeline("ner", model=model, tokenizer=tokenizer)
anonymizer = Anonymizer(config, classifier)

@mcp.tool()
def anonymize(text: str) -> str:
    return anonymizer.anonymize(text)

if __name__ == "__main__":
    mcp.run()
