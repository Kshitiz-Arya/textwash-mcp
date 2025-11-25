import torch
from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from .config import Config
from .anonymizer import Anonymizer
from .utils import get_available_entities

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
def analyze_pii(text: str, language: str = "en") -> str:
    anonymizer = get_anonymizer(language)
    entities = anonymizer.analyze(text)
    if not entities: return "No PII found."
    return str([e["word"] + ": " + e["entity"] for e in entities])

@mcp.tool()
def list_supported_entity_types(language: str = "en") -> str:
    config = Config(language=language)
    entities = get_available_entities(config.path_to_model)
    return ", ".join(entities)

@mcp.tool()
def analyze_pii(text: str, language: str = "en") -> str:
    anonymizer = get_anonymizer(language)
    entities = anonymizer.analyze(text)
    if not entities: return "No PII found."
    return str([e["word"] + ": " + e["entity"] for e in entities])

@mcp.tool()
def anonymize_text(text: str, language: str = "en", mode: str = "standard", restrict_to_entities: list[str] = None) -> str:
    anonymizer = get_anonymizer(language)
    return anonymizer.anonymize(text, selected_entities=restrict_to_entities)

if __name__ == "__main__":
    mcp.run()

@mcp.tool()
def anonymize_and_generate_key(text: str, language: str = "en") -> str:
    import json
    anonymizer = get_anonymizer(language)
    text_out, mapping = anonymizer.anonymize(text, strategy="standard", return_mapping=True)
    return json.dumps({"anonymized_text": text_out, "key": mapping}, indent=2)

@mcp.tool()
def anonymize_file(input_path: str, output_path: str, language: str = "en", mode: str = "standard") -> str:
    import os
    if not os.path.exists(input_path): return "Input file not found."
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    anonymizer = get_anonymizer(language)
    result = anonymizer.anonymize(content, strategy=mode)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
        
    return f"Processed {input_path} -> {output_path}"
