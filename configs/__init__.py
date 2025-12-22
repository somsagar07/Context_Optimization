"""
Configuration loader for Context_Opt.

Usage:
    from configs import load_config
    cfg = load_config("multi_step")  # or "single_step"
    
    # Access settings
    print(cfg.ENV_MODE)
    print(cfg.LEARNING_RATE)
"""
import importlib
import sys
from types import ModuleType


def load_config(config_name: str) -> ModuleType:
    """
    Load a configuration module by name.
    
    Args:
        config_name: Either "single_step" or "multi_step"
        
    Returns:
        The configuration module with all settings as attributes
    """
    valid_configs = ["single_step", "multi_step"]
    
    if config_name not in valid_configs:
        raise ValueError(f"Invalid config: {config_name}. Must be one of {valid_configs}")
    
    # Import the specific config module
    module = importlib.import_module(f"configs.{config_name}")
    
    return module


def get_config_from_args():
    """
    Parse command line arguments to get config.
    Useful for scripts that want to auto-detect config from CLI.
    
    Returns:
        Tuple of (config_module, config_name)
    """
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default="multi_step",
                        choices=["single_step", "multi_step"],
                        help="Configuration to use")
    args, _ = parser.parse_known_args()
    
    return load_config(args.config), args.config


# For backwards compatibility, also expose base settings
from configs.base import *

