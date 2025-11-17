# handle loading environment variables from the .env file
# call this module at the start of each script that needs environment variables
# call like this:
#   from src.config.env_loader import *
#
# an example .env file is provided as .env_template
#
# environmental variables are then loaded with os.getenv("VAR_NAME")
# you can also use the helper function get_env_var("VAR_NAME", default_value). If the variable is not found, it returns default_value.
# e.g., DATA_HOME = get_env_var("DATA_HOME", "/default/path/to/data")

from pathlib import Path
from dotenv import load_dotenv
import os

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=False)

def get_env_var(var_name, default=None):
    """
    Get an environment variable or return a default value.
    """
    return os.getenv(var_name, default)