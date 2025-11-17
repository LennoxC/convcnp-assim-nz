# handle loading environment variables from the .env file
# call this module at the start of each script that needs environment variables
# call like this:
#   from src.config.env_loader import *
#
# an example .env file is provided as .env_template
#
# environmental variables are then loaded with os.getenv("VAR_NAME")


from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=False)