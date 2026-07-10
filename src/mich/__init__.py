from importlib.resources import files
from pathlib import Path

REPO_ROOT = Path(str(files("mich"))).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
