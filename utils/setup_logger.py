import logging
import os
import tomllib
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# -------------------------
# Load config
# -------------------------
config_path = os.path.join(os.path.dirname(__file__), "config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)

LOG_LEVEL = getattr(
    logging,
    config.get("log", {}).get("level", "INFO").upper(),
    logging.INFO
)

# -------------------------
# Color mapping
# -------------------------
LOG_COLORS = {
    "DEBUG": Fore.BLUE,
    "INFO": Fore.GREEN,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}

# -------------------------
# Formatter
# -------------------------
class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, "")
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        return (
            f"{color}[{record.levelname}] "
            f"{time_str} "
            f"[{record.name}] "   # ← THIS is what you want
            f"{record.getMessage()}"
            f"{Style.RESET_ALL}"
        )

# -------------------------
# Root logger setup (once)
# -------------------------
def _setup_root_logger():
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)

# -------------------------
# Public API
# -------------------------
def get_logger(name: str):
    _setup_root_logger()
    return logging.getLogger(name)