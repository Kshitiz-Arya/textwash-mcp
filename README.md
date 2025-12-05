# TextWash MCP - PII Anonymization Server

A powerful Model Context Protocol (MCP) server that automatically detects and anonymizes Personally Identifiable Information (PII) in text documents using advanced machine learning models.

## Features

- **Multi-language Support**: Dutch (BERT) and English (RoBERTa) models
- **Smart PII Detection**: ML-powered entity recognition for names, locations, organizations, etc.
- **Flexible Anonymization**: Choose between numbered placeholders (`PERSON_1`) or redaction (`[REDACTED]`)
- **Selective Anonymization**: Target specific entity types while leaving others intact
- **Heuristic Enhancement**: Catches patterns ML models might miss (numbers, pronouns, titles)
- **Reversible Mapping**: Generate keys to map back from anonymized text to original
- **File Processing**: Batch anonymize entire documents
- **High Performance**: Model caching and GPU acceleration support

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9.0+
- Git

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Setup

### 1. Data Directory Structure

Create a `data/` directory with the following structure:

```
data/
├── months.txt          # Month names for temporal detection
├── written_numbers.txt # Written numbers (one, two, three, etc.)
├── nl/                # Dutch BERT model files
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── en/                # English RoBERTa model files
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

### 2. Model Files

Place your trained BERT (Dutch) and RoBERTa (English) models in the respective language directories. The models should be in Hugging Face format with:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- Tokenizer files (`tokenizer.json`, `vocab.txt`, etc.)

### 3. Reference Data Files

Create reference files in the `data/` directory:

**months.txt**:
```
January
February
March
...
December
```

**written_numbers.txt**:
```
zero
one
two
three
...
```

## Usage

### Running the MCP Server

```bash
python -m src.server
```

The server will start and expose MCP tools for text anonymization.

### Basic Text Anonymization

```python
from src.anonymizer import Anonymizer
from src.config import Config

# Initialize for English
config = Config(language="en")
# ... load classifier ...
anonymizer = Anonymizer(config, classifier)

# Anonymize text
text = "John Smith from New York called about order 12345"
result = anonymizer.anonymize(text)
print(result)
# Output: "PERSON_1 from LOCATION_1 called about order NUMERIC_1"
```

## API Reference

### MCP Tools

The server exposes the following tools via the Model Context Protocol:

#### `analyze_pii(text, language="en")`
Detect PII entities without anonymizing.

**Parameters:**
- `text` (str): Input text to analyze
- `language` (str): Language model to use ("en" or "nl")

**Returns:** String listing detected entities and their types

**Example:**
```
Input: "Dr. Sarah works at Microsoft"
Output: "Dr.: TITLE, Sarah: PERSON, Microsoft: ORGANIZATION"
```

#### `list_supported_entity_types(language="en")`
Get all entity types the model can detect.

**Parameters:**
- `language` (str): Language model to query

**Returns:** Comma-separated list of entity types

#### `anonymize_text(text, language="en", mode="standard", restrict_to_entities=None)`
Anonymize PII in text with optional filtering.

**Parameters:**
- `text` (str): Input text to anonymize
- `language` (str): Language model to use
- `mode` (str): "standard" (numbered) or "redact" ([REDACTED])
- `restrict_to_entities` (list): Optional list of entity types to anonymize

**Example:**
```python
# Anonymize only person names
anonymize_text("John works at Microsoft", restrict_to_entities=["PERSON"])
# Output: "PERSON_1 works at Microsoft"
```

#### `anonymize_and_generate_key(text, language="en")`
Anonymize text and return mapping for reverse operation.

**Returns:** JSON with anonymized text and mapping dictionary

#### `anonymize_file(input_path, output_path, language="en", mode="standard")`
Process entire files for anonymization.

### Core Classes

#### `Anonymizer`
Main anonymization engine.

**Methods:**
- `analyze(text)` - Detect PII entities
- `anonymize(text, selected_entities=None, strategy="standard", return_mapping=False)` - Anonymize text
- `replace_heuristics(text)` - Apply rule-based anonymization

#### `Config`
Configuration management for language-specific setup.

**Parameters:**
- `language` (str): "nl" for Dutch or "en" for English

## Examples

### Example 1: Basic Anonymization

```python
from src.config import Config
from src.anonymizer import Anonymizer

# Setup (assuming you have a classifier)
config = Config("en")
anonymizer = Anonymizer(config, classifier)

# Input text
text = """
Hi John,

Sarah from Microsoft called about the Boston project. 
Her number is 555-123-4567. Please call back by Monday.

Thanks,
Dr. Smith
"""

# Anonymize
result = anonymizer.anonymize(text)
print(result)
```

**Output:**
```
Hi PERSON_1,

PERSON_2 from ORGANIZATION_1 called about the LOCATION_1 project. 
Her number is NUMERIC_1-NUMERIC_2-NUMERIC_3. Please call back by Monday.

Thanks,
TITLE PERSON_3
```

### Example 2: Selective Anonymization

```python
# Only anonymize person names, keep everything else
result = anonymizer.anonymize(
    text, 
    selected_entities=["PERSON"]
)
```

### Example 3: Redaction Mode

```python
# Complete redaction instead of numbered placeholders
result = anonymizer.anonymize(
    text, 
    strategy="redact"
)
print(result)
# Output: "[REDACTED] from [REDACTED] called about [REDACTED]..."
```

### Example 4: With Mapping

```python
# Get anonymized text and reverse mapping
anonymized, mapping = anonymizer.anonymize(
    text, 
    return_mapping=True
)

print("Mapping:", mapping)
# Output: {"John": "PERSON_1", "Sarah": "PERSON_2", ...}
```

### Key Components

- **`anonymizer.py`**: Core PII detection and anonymization engine
- **`utils.py`**: Token processing and model configuration utilities  
- **`config.py`**: Language-specific configuration and validation
- **`server.py`**: MCP server with caching and tool definitions

## Configuration

### Supported Languages

- **English (`en`)**: Uses RoBERTa model for entity recognition
- **Dutch (`nl`)**: Uses BERT model for entity recognition

### Entity Types

The system can detect various entity types depending on the model training:
- `PERSON` - Person names
- `LOCATION` - Geographic locations  
- `ORGANIZATION` - Company/organization names
- `TITLE` - Professional titles
- `EMAIL` - Email addresses
- `PHONE` - Phone numbers
- And more based on your model training

### Anonymization Strategies

1. **Standard**: Numbered placeholders (`PERSON_1`, `LOCATION_1`, etc.)
2. **Redact**: Generic redaction markers (`[REDACTED]`)

## Use Cases

- **Document Sharing**: Remove PII from customer feedback, support tickets
- **Data Analysis**: Anonymize datasets while preserving structure for analytics
- **Legal Compliance**: GDPR, HIPAA compliance for document processing
- **Research**: Anonymize sensitive documents for academic research
- **Testing**: Create anonymized test datasets from production data

## Performance Features

- **Model Caching**: Loaded models are cached to avoid reloading
- **GPU Support**: Automatic GPU detection and usage when available
- **Batch Processing**: Efficient processing of large documents
- **Memory Efficient**: Streaming processing for large files

## Privacy & Security

- **Local Processing**: All anonymization happens locally, no data sent to external services
- **Reversible**: Optional mapping generation allows controlled de-anonymization
- **Configurable**: Fine-grained control over what gets anonymized
- **Audit Trail**: Track what entities were detected and replaced

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face Transformers library for model infrastructure
- Model Context Protocol (MCP) for the server framework
- PyTorch for machine learning capabilities

---

**Note**: This project requires trained NER models for each supported language. Model files are not included in the repository and must be obtained separately.