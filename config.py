"""
Global configuration for receipt counting system.
Contains class distribution settings and calibration parameters that can be updated
as the real-world distribution changes.
"""

import os
import json
import torch

# Default class distribution - can be overridden by environment or config file
DEFAULT_CLASS_DISTRIBUTION = [0.4, 0.2, 0.2, 0.1, 0.1]

# Calibration factors will be derived from class distribution automatically


class Config:
    """Global configuration singleton for receipt counting system."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Initialize with defaults
        self.class_distribution = DEFAULT_CLASS_DISTRIBUTION.copy()
        
        # Load from config file if it exists, silently during initialization
        config_path = os.environ.get("RECEIPT_CONFIG_PATH", "receipt_config.json")
        if os.path.exists(config_path):
            self.load_from_file(config_path, silent=True)

        # Environment variables override file settings
        self._load_from_env()

        # Calculate derived values including calibration factors
        self._update_derived_values()

        self._initialized = True

    def _load_from_env(self):
        """Load configuration from environment variables if present."""
        # Format for env vars: RECEIPT_CLASS_DIST="0.3,0.2,0.2,0.1,0.1,0.1"
        if "RECEIPT_CLASS_DIST" in os.environ:
            try:
                dist_str = os.environ["RECEIPT_CLASS_DIST"]
                dist = [float(x) for x in dist_str.split(",")]
                # Validate
                if len(dist) == 5 and abs(sum(dist) - 1.0) < 0.01:
                    self.class_distribution = dist
            except Exception as e:
                print(f"Warning: Invalid RECEIPT_CLASS_DIST format: {e}")

        # We no longer need to load calibration factors from environment variables
        # as they will be automatically derived from class distribution

    def load_from_file(self, config_path, silent=False):
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
            silent: If True, don't print a message when loading config
        """
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Update distribution if valid
            if "class_distribution" in config_data:
                dist = config_data["class_distribution"]
                if len(dist) == 5 and abs(sum(dist) - 1.0) < 0.01:
                    self.class_distribution = dist

            # We no longer need to load calibration factors from config file
            # as they will be automatically derived from class distribution

            self._update_derived_values()
            
            if not silent:
                print(f"Loaded configuration from {config_path}")

        except Exception as e:
            if not silent:
                print(f"Error loading config from {config_path}: {e}")

    def save_to_file(self, config_path):
        """Save current configuration to a JSON file."""
        config_data = {
            "class_distribution": self.class_distribution,
            # We include the derived calibration factors as information only
            "derived_calibration_factors": self.calibration_factors,
        }

        try:
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config to {config_path}: {e}")

    def _update_derived_values(self):
        """Update any derived configuration values."""
        # Calculate inverse weights for CrossEntropyLoss
        self.inverse_weights = [
            1.0 / w if w > 0 else 1.0 for w in self.class_distribution
        ]
        sum_inverse = sum(self.inverse_weights)
        self.normalized_weights = [w / sum_inverse for w in self.inverse_weights]
        self.scaled_weights = [
            w * len(self.class_distribution) for w in self.normalized_weights
        ]  # Scale by num classes
        
        # Always derive calibration factors from class distribution using a principled approach
        # For calibration factors, we need to counteract the class weighting during inference
        # A principled approach is to use:
        # calibration_factor = prior_probability * sqrt(reference_probability / prior_probability)
        # This balances the influence of the prior while adding compensation for minority classes
        reference_prob = 1.0 / len(self.class_distribution)  # Equal distribution
        self.calibration_factors = [
            prior * ((reference_prob / prior) ** 0.5) if prior > 0 else 1.0 
            for prior in self.class_distribution
        ]
        
        # Normalize calibration factors for better comparison
        max_cal = max(self.calibration_factors)
        self.calibration_factors = [cal / max_cal for cal in self.calibration_factors]

    def get_class_weights_tensor(self, device=None):
        """Get class weights as a tensor for CrossEntropyLoss."""
        weights = torch.tensor(self.scaled_weights)
        if device:
            weights = weights.to(device)
        return weights

    def get_calibration_tensor(self, device=None):
        """Get calibration factors as a tensor for inference."""
        cal = torch.tensor(self.calibration_factors)
        if device:
            cal = cal.to(device)
        return cal

    def get_class_prior_tensor(self, device=None):
        """Get class distribution as a tensor for calibration."""
        prior = torch.tensor(self.class_distribution)
        if device:
            prior = prior.to(device)
        return prior

    def update_class_distribution(self, new_distribution):
        """Update the class distribution and recalculate derived values."""
        if len(new_distribution) != 5 or abs(sum(new_distribution) - 1.0) > 0.01:
            raise ValueError(
                "Class distribution must have 5 values that sum to approximately 1.0"
            )

        self.class_distribution = new_distribution
        self._update_derived_values()

    # We no longer need a method to manually update calibration factors
    # as they are automatically derived from class distribution


    def explain_calibration(self):
        """Explain how calibration factors are derived from class distribution."""
        reference_prob = 1.0 / len(self.class_distribution)
        explanation = {
            "class_distribution": self.class_distribution,
            "reference_probability": reference_prob,
            "derivation_steps": []
        }
        
        # Calculate the raw factors before normalization
        raw_factors = []
        for i, prior in enumerate(self.class_distribution):
            if prior > 0:
                sqrt_term = (reference_prob / prior) ** 0.5
                raw_factor = prior * sqrt_term
            else:
                raw_factor = 1.0
                
            raw_factors.append(raw_factor)
            
            step = {
                "class": i,
                "prior_probability": prior,
                "sqrt_term": sqrt_term if prior > 0 else "N/A",
                "raw_factor": raw_factor
            }
            explanation["derivation_steps"].append(step)
        
        # Add normalization step
        max_raw = max(raw_factors)
        explanation["max_raw_factor"] = max_raw
        explanation["normalized_factors"] = self.calibration_factors
        
        return explanation


# Singleton instance for easy import
config = Config()


def get_config():
    """Get the global configuration instance."""
    return config
