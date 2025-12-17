"""
Utility for loading API keys and configurations
"""

import os, yaml

def load_api_keys():
    """Load all API keys from configs/api_keys.yaml"""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    path = os.path.join(root_dir, "configs", "api_keys.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
