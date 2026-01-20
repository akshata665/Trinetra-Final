import logging

# Create logger
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)

# Console handler
console = logging.StreamHandler()
console.setFormatter(formatter)

# Avoid duplicate handlers if module re-imports
if not logger.handlers:
    logger.addHandler(console)
