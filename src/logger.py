import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(filename=LOG_FILE_PATH,level=logging.INFO,
                    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s ")

if __name__=="__main__":
    logging.info("This is a test message")
    logging.error("This is an error message")
    logging.warning("This is a warning message")
    logging.debug("This is a debug message")
    logging.critical("This is a critical message")
    logging.exception("This is an exception message")