import logging
from dotenv import load_dotenv

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

# 환경 변수 로드
load_dotenv()

