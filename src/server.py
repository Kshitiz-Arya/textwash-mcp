"""
MCP (Model Context Protocol) server for text anonymization.

Provides tools for PII detection and anonymization via MCP interface.
Supports both Dutch and English models with caching for performance.
"""
import torch
from mcp.server.fastmcp import FastMCP
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from .config import Config
from .anonymizer import Anonymizer
from .utils import get_available_entities

# Initialize MCP server
mcp = FastMCP("Textwash Anonymizer")

# Cache loaded models to avoid reloading on each request
_MODEL_CACHE = {}

def get_anonymizer(language: str):
    """Get or create an Anonymizer instance for the specified language.
    
    Uses caching to avoid reloading models. Automatically detects GPU availability.
    
    Args:
        language: Language code ('en' or 'nl')
        
    Returns:
        Anonymizer instance ready for use
    """
    # Return cached instance if available
    if language in _MODEL_CACHE:
        return _MODEL_CACHE[language]

    print(f"Loading model for {language}...")
    config = Config(language=language)
    
    # Use GPU if available, otherwise fallback to CPU
    device = 0 if torch.cuda.is_available() else -1
    
    # Load tokenizer and model from local files
    tokenizer = AutoTokenizer.from_pretrained(config.path_to_model)
    model = AutoModelForTokenClassification.from_pretrained(config.path_to_model)
    classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    
    # Create and cache anonymizer instance
    anonymizer = Anonymizer(config, classifier)
    _MODEL_CACHE[language] = anonymizer
    return anonymizer

# MCP Tools - Functions exposed via the Model Context Protocol

@mcp.tool()
def analyze_pii(text: str, language: str = "en") -> str:
    """Detect PII entities in text without anonymizing.
    
    Returns a human-readable list of detected entities and their types.
    """
    anonymizer = get_anonymizer(language)
    entities = anonymizer.analyze(text)
    if not entities: 
        return "No PII found."
    return str([e["word"] + ": " + e["entity"] for e in entities])

@mcp.tool()
def list_supported_entity_types(language: str = "en") -> str:
    """List all entity types that can be detected by the specified language model."""
    config = Config(language=language)
    entities = get_available_entities(config.path_to_model)
    return ", ".join(entities)

@mcp.tool() 
def anonymize_text(text: str, language: str = "en", mode: str = "standard", restrict_to_entities: list[str] = None) -> str:
    """Anonymize PII in text with optional entity type filtering.
    
    Args:
        text: Input text to anonymize
        language: Language model to use ('en' or 'nl')
        mode: Anonymization strategy ('standard' or 'redact')
        restrict_to_entities: List of entity types to anonymize (None = all)
    """
    anonymizer = get_anonymizer(language)
    return anonymizer.anonymize(text, selected_entities=restrict_to_entities)

@mcp.tool()
def anonymize_and_generate_key(text: str, language: str = "en") -> str:
    """Anonymize text and return both result and mapping as JSON.
    
    Useful for creating reversible anonymization where you need to map
    back from anonymized placeholders to original values.
    """
    import json
    anonymizer = get_anonymizer(language)
    text_out, mapping = anonymizer.anonymize(text, strategy="standard", return_mapping=True)
    return json.dumps({"anonymized_text": text_out, "key": mapping}, indent=2)

@mcp.tool()
def anonymize_file(input_path: str, output_path: str, language: str = "en", mode: str = "standard") -> str:
    """Process an entire file for anonymization.
    
    Reads from input_path, anonymizes content, and writes to output_path.
    Creates output directory if it doesn't exist.
    
    Args:
        input_path: Path to file to anonymize
        output_path: Where to save anonymized result
        language: Language model to use
        mode: Anonymization strategy
        
    Returns:
        Status message indicating success or failure
    """
    import os
    
    # Validate input file exists
    if not os.path.exists(input_path): 
        return "Input file not found."
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Anonymize content
    anonymizer = get_anonymizer(language)
    result = anonymizer.anonymize(content, strategy=mode)
    
    # Ensure output directory exists and write result
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result)
        
    return f"Processed {input_path} -> {output_path}"

# Run MCP server when executed directly
if __name__ == "__main__":
    mcp.run()
