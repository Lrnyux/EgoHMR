import logging
import time
import os








class AverageMeter():
    def __init__(self):
        self.val = 0
        self.count = 0
        self.sum = 0
        self.ave = 0

    def update(self, val, num=1):
        self.count = self.count + num
        self.val = val
        self.sum = self.sum + num * val
        self.ave = self.sum / self.count if self.count != 0 else 0.0


class CustomFormatter(logging.Formatter):
    DATE = '\033[94m'
    GREEN = '\033[92m'
    WHITE = '\033[0m'
    WARNING = '\033[93m'
    RED = '\033[91m'

    def __init__(self):
        orig_fmt = "%(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        super().__init__(orig_fmt, datefmt)

    def format(self, record):
        color = self.WHITE
        if record.levelno == logging.INFO:
            color = self.GREEN
        if record.levelno == logging.WARN:
            color = self.WARNING
        if record.levelno == logging.ERROR:
            color = self.RED
        self._style._fmt = "{}%(asctime)s {}[%(levelname)s]{} {}: %(message)s".format(
            self.DATE, color, self.DATE, self.WHITE)
        return logging.Formatter.format(self, record)


class ConsoleLogger():
    def __init__(self, training_type, phase='train'):
        super().__init__()
        self._logger = logging.getLogger(training_type)
        self._logger.setLevel(logging.INFO)
        formatter = CustomFormatter()
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        console_log.setFormatter(formatter)
        self._logger.addHandler(console_log)
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.logfile_dir = os.path.join('experiments/', training_type, time_str)
        os.makedirs(self.logfile_dir)
        logfile = os.path.join(self.logfile_dir, f'{phase}.log')
        file_log = logging.FileHandler(logfile, mode='a')
        file_log.setLevel(logging.INFO)
        file_log.setFormatter(formatter)
        self._logger.addHandler(file_log)

    def info(self, *args, **kwargs):
        """info"""
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        """warning"""
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        """error"""
        self._logger.error(*args, **kwargs)
        exit(-1)

    def getLogFolder(self):
        return self.logfile_dir