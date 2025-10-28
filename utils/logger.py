from datetime import datetime
import threading
import os

class Logger:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, filepath=None):
        with cls._lock:
            if cls._instance is None:
                if filepath is None:
                    raise ValueError("Logger must be initialized with a filepath the first time.")
                cls._instance = super().__new__(cls)
                cls._instance.filepath = filepath
            return cls._instance

    def save(self, message):
        if os.path.exists(self.filepath) is False:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # get current date for log file naming
        year, month, day = now.split(" ")[0].split("-")
        with open(f"{self.filepath}/{year}-{month}-{day}.log", 'a', encoding='utf-8') as f:
            f.write(f"{now} - {message}\n")

    def log(self, message):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} - {message}")
