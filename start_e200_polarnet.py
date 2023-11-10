import warnings
import torch
from tools.dataset import PolarNetDataset
from tools.train_process import PolarNetTrainProcess
from model.polarnet_family import polarnet18, polarnet34
from Config import PolarNetConfig
from tools.process_manager import ProcessManager
from tools.utils_pack import str_size

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    PolarNetConfig.prior_k: torch.Tensor = torch.ones((3, 8, 3), dtype=torch.float)
    # Prior Knowledge # This is a unstale feature.
    PolarNetConfig.prior_k[1, :, :] = 2.
    PolarNetConfig.prior_k[0, 5, 1] = 2.
    PolarNetConfig.prior_k[0, 6, 1] = 2.
    PolarNetConfig.prior_k[0, 3, 2] = 1.5
    PolarNetConfig.prior_k[0, 4, 2] = 1.5
    PolarNetConfig.prior_k[2, 0, 1] = 1.5
    PolarNetConfig.prior_k[2, 7, 1] = 1.5
    PolarNetConfig.prior_k[2, 3, 1] = 1.5
    PolarNetConfig.prior_k[2, 4, 1] = 1.5
    PolarNetConfig.prior_k[2, 1, 2] = 1.5
    PolarNetConfig.prior_k[2, 2, 2] = 1.5
    PolarNetConfig.prior_k -= torch.min(PolarNetConfig.prior_k)
    PolarNetConfig.prior_k /= torch.max(PolarNetConfig.prior_k)

    PolarNetConfig.n_fold = [1, 2, 3, 4, 5]
    PolarNetConfig.batch_size = 28
    PolarNetConfig.weight = [.11, .21]
    PolarNetConfig.input_size = str_size(PolarNetConfig.resize_to_ori)
    ProcessManager(
        devices="8,9",
        env_pre="NoPt_PolarNets18_Polar_Adaptive",
        cfg=PolarNetConfig,
        network=polarnet18,
        dataset=PolarNetDataset,
        train_process=PolarNetTrainProcess,
        is_cat=False,
    ).run()

    PolarNetConfig.n_fold = [1, 2, 3, 4, 5]
    PolarNetConfig.batch_size = 28
    PolarNetConfig.weight = [.11, .21]
    PolarNetConfig.input_size = str_size(PolarNetConfig.resize_to_ori)
    ProcessManager(
        devices="0,1",
        env_pre="NoPt_PolarNets34_Polar_Adaptive",
        cfg=PolarNetConfig,
        network=polarnet34,
        dataset=PolarNetDataset,
        train_process=PolarNetTrainProcess,
        is_cat=False,
    ).run()

