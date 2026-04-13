from dotenv import find_dotenv
from pathlib import Path

env_path = Path(find_dotenv())
ROOT_DIR = env_path.parent