import sys
import logging
from src.logger import logging

# Function to extract error details
def error_message_detail(error, error_detail: sys):
    """
    Constructs a detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()  # Get exception traceback
    file_name = exc_tb.tb_frame.f_code.co_filename  # File where the error occurred
    error_message = "Error occurred in python script: [{0}] at line number [{1}] with message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

# Custom exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

# Min block to simulate exception
if __name__ == "__main__":
    try:
        a = 1 / 0  # Deliberate division by zero to trigger an exception
    except Exception as e:
        logging.basicConfig(filename="error.log", level=logging.INFO)  # Log errors to a file
        logging.info("Logging has started")
        raise CustomException(e, sys)
