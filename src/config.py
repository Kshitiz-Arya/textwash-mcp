"""
Configuration management for the TextWash anonymizer.

Handles language-specific model selection and file path setup.
"""
import os
from pathlib import Path

class Config:
    """Configuration class for anonymization setup.
    
    Manages file paths and model selection based on target language.
    Currently supports Dutch (nl) and English (en).
    """
    
    def __init__(self, language: str):
        """Initialize configuration for specified language.
        
        Args:
            language: 'nl' for Dutch or 'en' for English
            
        Raises:
            ValueError: If language is not supported
            FileNotFoundError: If model files are not found
        """
        # Validate language selection
        if language not in ["nl", "en"]:
            raise ValueError(f"Invalid language '{language}' specified")
            
        # Set up directory paths relative to project root
        base_dir = Path(__file__).parent.parent.resolve()
        data_dir = base_dir / "data"

        # Configure file paths for reference data and models
        self.path_to_months_file = str(data_dir / "months.txt")  # Month names for temporal detection
        self.path_to_written_numbers_file = str(data_dir / "written_numbers.txt")  # Written numbers
        self.path_to_model = str(data_dir / language)  # Language-specific model directory
        
        # Select appropriate model type based on language
        self.model_type = "bert" if language == "nl" else "roberta"
        
        # Verify model exists before proceeding
        if not os.path.exists(self.path_to_model):
            raise FileNotFoundError(f"Model not found at {self.path_to_model}")
