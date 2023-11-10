import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import socket
from tools.train_process import TrainProcessBase


# function implementation 
def push_deer(key: str, text: str):
    url = "https://api2.pushdeer.com/message/push?pushkey=" + str(key) + "&text=" + str(text)
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.get(url)
    return


# object-oriented implementation
class PushDeer():

    def __init__(self, key: str) -> None:
        self.key: str = str(key)
        self.session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.hostname = str(socket.gethostname())

    def push(self, text: str):
        url = "https://api2.pushdeer.com/message/push?pushkey=" + self.key + "&text=" + str(text)
        try:
            self.session.get(url)
        except:
            _ = 1
        return

    def push_exit(self, info: str = ""):
        endl = "%0A"
        date_time = time.strftime("%A, %B %d, %Y, %H:%M:%S", time.localtime()) + endl
        text = self.hostname + " " + "程序结束" + endl + date_time + endl
        text += info
        self.push(text.replace("&", "and"))
        return

    def push_program_exit(self, ENV: str = "", avg_acc: float = -1, avg_auc: float = -1):
        try:
            info = ""
            info = "ENV = {}".format(str(ENV))
            info += "%0AAVG_ACC = {} ".format(str(avg_acc))
            info += "%0AAVG_AUC = {}".format(str(avg_auc))
            self.push_exit(info)
        except:
            _ = 1
        return

    def push_model_exit(self, train_process: TrainProcessBase, ENV: str = "", Fold=-1):
        try:
            ACC_MAX = train_process.record_best_acc.MAX_ACC
            AUC_MAX = train_process.record_best_auc.MAX_AUC
            ACC_Ep = train_process.record_best_acc.Epoch
            AUC_Ep = train_process.record_best_auc.Epoch
            ACC_MAX = format(ACC_MAX, '.3f')
            AUC_MAX = format(AUC_MAX, '.3f')

            info = ""
            info = "ENV = {} -> Fold {}".format(str(ENV), str(Fold))
            info += "%0AACC = {} @Epoch {}".format(str(ACC_MAX), str(ACC_Ep))
            info += "%0AAUC = {} @Epoch {}".format(str(AUC_MAX), str(AUC_Ep))
            self.push_exit(info)
        except:
            _ = 1
        return