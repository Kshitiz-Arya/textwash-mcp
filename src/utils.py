"""
Utility functions for processing ML model outputs and configuration.
"""
import json

def decode_outputs(predicted_labels, model_type="bert"):
    """Fix tokenization artifacts from BERT/RoBERTa models.
    
    BERT uses ## to indicate subword tokens that should be merged.
    RoBERTa uses Ġ to indicate word starts, absence means continuation.
    
    Args:
        predicted_labels: Raw predictions from transformers pipeline
        model_type: 'bert' or 'roberta' to handle different tokenization
    
    Returns:
        List of entities with properly reconstructed words
    """
    entities = []
    shift_idx = 2 if model_type == "bert" else 0  # Characters to skip when merging tokens

    for _, elem in enumerate(predicted_labels):
        attach = False  # Whether this token should be merged with previous
        
        # Determine if token is a continuation based on model type
        if model_type == "bert" and elem["word"].startswith("##"):
            attach = True
        elif model_type == "roberta" and not elem["word"].startswith("Ġ"):
            attach = True

        if attach and entities:
            # Merge with previous token
            entities[-1]["word"] += elem["word"][shift_idx:]
            entities[-1]["end"] = elem["end"]  # Update end position
        else:
            # Start new entity
            entities.append({
                "word": elem["word"],
                "start": elem["start"],
                "end": elem["end"],
                "entity": elem["entity"],
            })
            
    # Clean up RoBERTa word-start markers
    if model_type == "roberta":
        for elem in entities:
            if elem["word"].startswith("Ġ"):
                elem["word"] = elem["word"][1:]  # Remove Ġ prefix
    return entities

def get_available_entities(model_path):
    """Extract supported entity types from a trained model's configuration.
    
    Args:
        model_path: Path to the model directory containing config.json
    
    Returns:
        Sorted list of entity types the model can detect (excludes generic labels)
    """
    import os
    import json
    
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path): 
        return []
        
    # Read model configuration to get label mapping
    with open(config_path, "r") as f:
        label_map = json.load(f)["id2label"]
    
    # Filter out generic labels, return sorted entity types
    available = sorted(list(set(label_map.values()) - {"NONE", "PAD", "O"}))
    return available
