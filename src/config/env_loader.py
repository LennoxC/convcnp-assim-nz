# handle loading environment variables from the .env file
# call this module at the start of each script that needs environment variables
# call like this:
#   from src.config.env_loader import *
#
# an example .env file is provided as .env_template
#
# environmental variables are then loaded with os.getenv("VAR_NAME")
# you can also use the helper function get_env_var("VAR_NAME", default_value). If the variable is not found, it returns default_value (which is None by default).
# e.g., DATA_HOME = get_env_var("DATA_HOME", "/default/path/to/data")

from pathlib import Path
from dotenv import load_dotenv
import os

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=False)

def get_env_var(var_name, default=None, *, return_default_flag=False):
    """
    Get an environment variable or return a default value.

    - By default (when `return_used_default` is False) this returns the value only.
    - If `return_used_default` is True, it returns a tuple `(value, used_default)`
      where `used_default` is True if the returned value came from `default`.
      This may be useful for logging (e.g. to warn the user that a default was used).

    Note: this determines whether the default was used by checking whether
    the variable name exists in `os.environ`. An environment variable set to
    the empty string counts as present (so `used_default` will be False).
    """
    if var_name in os.environ:
        value = os.environ.get(var_name)
        used_default = False
    else:
        value = default
        used_default = True

    if return_default_flag:
        return value, used_default
    return value