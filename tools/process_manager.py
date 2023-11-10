import os
import torch
from torch import nn, optim
from tools.visualizer import RedundancyVisualizer
from tools.train_process import TrainProcessBase
from tools.push_tools import PushDeer
from tools.utils_pack import abbr_double, set_seed, AvgMax
from tools.draw_plot import AvgPlot
from Config import ConfigBase
import torch.utils.data as data


class ProcessManager():
    def __init__(
        self,
        devices,
        env_pre,
        cfg,
        network,
        dataset,
        train_process,
        criterion=nn.CrossEntropyLoss,
        is_cat: bool = True,
    ):
        self.CUDA_VISIBLE_DEVICES: str = devices
        self.env_pre: str = env_pre
        self.cfg: ConfigBase = cfg
        self.is_cat: bool = is_cat
        self.dataset: data.Dataset = dataset
        self.network: nn.Module = network
        self.train_process: TrainProcessBase = train_process
        self.criterion = criterion

    def run(self):
        cfg: ConfigBase = self.cfg
        os.environ["CUDA_VISIBLE_DEVICES"] = self.CUDA_VISIBLE_DEVICES
        env_name = "{}_{}_{}_{}_{}".format(self.env_pre, cfg.dataset_name, cfg.input_size, cfg.batch_size,
                                           abbr_double(cfg.layer_index_polar_ori))

        set_seed(98)
        data_path = "../../AD_problem/data/{}/".format(cfg.dataset_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vis = RedundancyVisualizer(env=env_name, servers=cfg.vis_servers, port=cfg.vis_port)
        save_path = os.path.join("./save/", env_name)
        push_deer = PushDeer(cfg.token)
        avg_plot = AvgPlot(to_=cfg.stop_epoch)
        avg_max = AvgMax(save_path=save_path)

        for fold_num in cfg.n_fold:
            print("\n交叉验证: ", fold_num)
            csv_path = os.path.join(data_path, "fiveFold", str(fold_num))
            # csv_path = os.path.join(data_path, "fourFold", str(fold_num))
            # csv_path = os.path.join(data_path, "threeFold", str(fold_num))
            cfg.prior_k = cfg.prior_k.to(device) if cfg.prior_k is not None else cfg.prior_k
            net_model = self.network(in_channel=cfg.in_channel, num_classes=cfg.num_classes,
                                     pretrained=cfg.pretrained).to(device)

            optimizer = optim.Adam(net_model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
            net_model = torch.nn.DataParallel(net_model).to(device)
            net_model = net_model.to(device)
            criterion = self.criterion(weight=torch.tensor(cfg.weight).to(device))
            # optimizer = optim.Adam(net_model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay)
            train_dataset: data.Dataset = self.dataset(
                data_path=data_path,
                csv_path=csv_path,
                config=cfg,
                is_training=True,
                is_cat=self.is_cat,
            )
            test_dataset: data.Dataset = self.dataset(
                data_path=data_path,
                csv_path=csv_path,
                config=cfg,
                is_training=False,
                is_cat=self.is_cat,
            )

            tp: TrainProcessBase = self.train_process(
                vis=vis,
                model=net_model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                save_path=save_path,
                config=cfg,
                env_name=env_name,
                fold_no=fold_num,
            )
            try:
                tp.train()
                avg_max.add(tp=tp)

            except:
                cfg.n_fold.append(fold_num)

            # avg_max.add(tp=tp)
            # push_deer.push_model_exit(ENV=env_name, Fold=fold_num, train_process=tp)

            del tp
            del test_dataset
            del train_dataset
            del criterion
            del optimizer
            del net_model

        avg_acc, avg_auc = avg_max.conclusion()
        push_deer.push_program_exit(ENV=env_name, avg_acc=avg_acc, avg_auc=avg_auc)
        avg_plot.vis_plot(result_list=[save_path], title="AVERAGE_PLOT", vis=vis)
