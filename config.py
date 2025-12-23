"""
Legacy config file for backwards compatibility.

DEPRECATED: Use configs/single_step.py or configs/multi_step.py instead.

This file imports from the default config (multi_step) for backwards compatibility
with any code that still does `import config`.
"""
# Import all settings from default config for backwards compatibility
from configs.multi_step import *

# Print deprecation warning when imported directly
import warnings
warnings.warn(
    "Importing from 'config' is deprecated. "
    "Use 'from configs import load_config' instead.",
    DeprecationWarning,
    stacklevel=2
)
