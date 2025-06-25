from loguru import logger
import os

os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add("logs/run.log", rotation="500 KB", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")
