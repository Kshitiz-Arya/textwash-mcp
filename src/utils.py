import json

def decode_outputs(predicted_labels, model_type="bert"):
    entities = []
    shift_idx = 2 if model_type == "bert" else 0

    for _, elem in enumerate(predicted_labels):
        attach = False
        if model_type == "bert" and elem["word"].startswith("##"):
            attach = True
        elif model_type == "roberta" and not elem["word"].startswith("Ġ"):
            attach = True

        if attach and entities:
            entities[-1]["word"] += elem["word"][shift_idx:]
            entities[-1]["end"] = elem["end"]
        else:
            entities.append({
                "word": elem["word"],
                "start": elem["start"],
                "end": elem["end"],
                "entity": elem["entity"],
            })
            
    if model_type == "roberta":
        for elem in entities:
            if elem["word"].startswith("Ġ"):
                elem["word"] = elem["word"][1:]
    return entities

def get_available_entities(model_path):
    import os
    import json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path): return []
    with open(config_path, "r") as f:
        label_map = json.load(f)["id2label"]
    available = sorted(list(set(label_map.values()) - {"NONE", "PAD", "O"}))
    return available
