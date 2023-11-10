import torch
from torch.utils.data import DataLoader
import progressbar
import os
from torch import nn
from pathlib import Path
import numpy as np

from tools.matrix import claMetrix
from tools.matrix import confuseMtrix
from tools.utils_pack import SaveCSV, RecordBestRes, RecordRes, SaveLastModel
from tools.utils_pack import draw_weight_cam, image2tensor, norm_npy


class TrainProcessBase():
    def __init__(
        self,
        vis,
        model,
        optimizer,
        criterion,
        device,
        train_dataset,
        test_dataset,
        save_path,
        config,
        env_name: str,
        class_num=2,
        fold_no=0,
    ):
        self.GLOBAL_WORKER_ID = None
        self.GLOBAL_SEED = 1
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.config = config
        self.env_name = env_name
        self.num_epochs = config.num_epochs
        self.stop_epoch = config.stop_epoch
        self.batch_size = config.batch_size
        self.fold_no = fold_no
        self.base_lr = config.base_lr
        self.vis = vis
        self.device = device
        self.epoch = 0
        self.save_path = save_path
        self.cam_save_epochs = config.cam_save_epochs
        self.is_use_cam = config.is_use_cam
        self.class_num = class_num
        self.record_best_auc = RecordBestRes(log_path=self.save_path, fold=self.fold_no, first="auc")
        self.record_best_acc = RecordBestRes(log_path=self.save_path, fold=self.fold_no, first="acc")
        self.record_best_kap = RecordBestRes(log_path=self.save_path, fold=self.fold_no, first="kappa")
        self.recorder = RecordRes(log_path=self.save_path, fold=self.fold_no)
        self.model_saver = SaveLastModel(save_path=self.save_path, fold=self.fold_no)

        self.ACC, self.AUC = 0.0, 0.0

        self.pbar = progressbar.ProgressBar(widgets=[
                                                    progressbar.Percentage(),
                                                    progressbar.Bar('>'),
                                                    progressbar.SimpleProgress(),
                                                    '|',
                                                    progressbar.Timer(),
                                                    '|',
                                                ],
                                            maxval=self.num_epochs)

    def test(self):
        data_loaderTest = DataLoader(self.test_dataset, batch_size=1, num_workers=0)
        self.model.eval()
        prediction_all = []
        gt_all = []
        step = 0
        weight_cam_all = [None, None, None]

        self.csv = SaveCSV(self.save_path, self.fold_no, self.epoch)

        for _, data in enumerate(data_loaderTest):
            step += 1

            _, outputs, labels, _, name, w = self.model_code_block(data, is_training=False)

            with torch.no_grad():
                weight_cam = [i.detach().cpu().clone() if i is not None else None
                              for i in w] if w is not None else weight_cam_all
                del w
                if step == 1:
                    prediction_all = outputs
                    gt_all = labels
                    weight_cam_all = weight_cam if weight_cam[0] is not None else weight_cam_all
                else:
                    prediction_all = torch.cat([prediction_all, outputs], dim=0)
                    gt_all = torch.cat([gt_all, labels], dim=0)

                    weight_cam_all[0] = torch.cat([weight_cam_all[0], weight_cam[0]],
                                                  dim=0) if weight_cam[0] is not None else weight_cam_all[0]
                    weight_cam_all[1] = torch.cat([weight_cam_all[1], weight_cam[1]],
                                                  dim=0) if weight_cam[1] is not None else weight_cam_all[1]
                    weight_cam_all[2] = torch.cat([weight_cam_all[2], weight_cam[2]],
                                                  dim=0) if weight_cam[2] is not None else weight_cam_all[2]

            self.__save2csv(outputs, labels, name)

        self.__show_save_weight(weight_cam_all)

        del self.csv
        return claMetrix(prediction_all, gt_all, class_num=self.class_num)

    def train(self):
        data_loaderTrain = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      drop_last=False,
                                      pin_memory=True,
                                      shuffle=False)
        self.pbar.start()

        # ACC, AUC, RE, PRE, F1, kappa, confMtrix = self.test()
        for epoch in range(self.stop_epoch):
            self.epoch = epoch + 1
            self.model.train(mode=True)
            for _, data in enumerate(data_loaderTrain):

                inputs, _, _, inputs_polar, _, _ = self.model_code_block(data, is_training=True)

                self.vis.img(name='Images_1', img_=inputs[0, :, :, :])
                if inputs_polar is not None:
                    self.vis.img(name='Images_2', img_=inputs_polar[0, :, :, :])

            self.__adjust_lr()

            if self.epoch % 1 == 0:
                ACC, AUC, RE, PRE, F1, kappa, confMtrix = self.test()
                self.ACC, self.AUC = ACC, AUC
                self.vis.plot('Test_AUC_' + str(self.fold_no), AUC)
                self.vis.plot('Test_ACC_' + str(self.fold_no), ACC)
                self.vis.plot('Test_kappa_' + str(self.fold_no), kappa)
                self.vis.plot('Test_F1_' + str(self.fold_no), F1)
                self.__confuseMtrix(confMtrix, is_vis_show=True)
                self.record_best_auc.update(model_now=self.model,
                                            epoch_now=self.epoch,
                                            acc_now=ACC,
                                            auc_now=AUC,
                                            kappa_now=kappa)
                self.record_best_acc.update(model_now=self.model,
                                            epoch_now=self.epoch,
                                            acc_now=ACC,
                                            auc_now=AUC,
                                            kappa_now=kappa)
                self.record_best_kap.update(model_now=self.model,
                                            epoch_now=self.epoch,
                                            acc_now=ACC,
                                            auc_now=AUC,
                                            kappa_now=kappa)
                self.recorder.update(acc_now=ACC, auc_now=AUC, kappa_now=kappa)
                self.model_saver.update(model_now=self.model, epoch_now=self.epoch)

            self.pbar.update(self.epoch)
        self.pbar.finish()
        self.record_best_acc.conclusion()
        self.record_best_auc.conclusion()
        self.record_best_kap.conclusion()
        self.recorder.close()

    def model_code_block(self, data, is_training: bool):
        raise NotImplementedError("TrainProcessBase.model_code_block: Not Implemented!")

    def __adjust_lr(self, power=0.9):
        lr = self.base_lr * (1 - float(self.epoch) / self.num_epochs)**power
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def __save2csv(self, output, labels, name):
        value = output[:, 1]
        threshhold = 0.5
        #大于 threshhold
        zero = torch.zeros_like(value)
        one = torch.ones_like(value)
        pred = torch.where(value > threshhold, one, zero)

        value, predicted, labels = value.cpu().detach().numpy(), pred.cpu().detach().numpy(), labels.cpu().detach().numpy()

        for i in range(len(name)):
            data = [name[i], value[i], int(predicted[i]), int(labels[i])]
            self.csv.write(data)

        return self

    def __show_save_weight(self, weight_cam_all):
        for i, wi in enumerate(weight_cam_all):
            if wi is not None:
                weight_cam_all[i] = torch.sum(wi, dim=0)

        if weight_cam_all[0] is not None:
            all = torch.cat(weight_cam_all, dim=0).detach().cpu().numpy().copy()

        for i, wi in enumerate(weight_cam_all):
            if wi is not None:
                wi = wi.detach().cpu().numpy().copy()
                wr = wi.copy()

                wr = norm_npy(wr)
                wi = norm_npy(wi, all)

                weight_path = os.path.join("./", "save/", self.env_name, "weight_cam", str(self.fold_no))
                Path(weight_path).mkdir(parents=True, exist_ok=True)
                np.save("{}/epoch_{}_{}_{}.npy".format(weight_path, self.epoch, 8, i), wi)
                np.save("{}/relative_epoch_{}_{}_{}.npy".format(weight_path, self.epoch, 8, i), wr)

                weight_cam_8 = draw_weight_cam(cam_weight=wi, mode=8)
                weight_cam_4 = draw_weight_cam(cam_weight=wi, mode=4)

                weight_cam_8r = draw_weight_cam(cam_weight=wr, mode=8)
                weight_cam_4r = draw_weight_cam(cam_weight=wr, mode=4)

                weight_cam_8.save("{}/epoch_{}_{}_{}.png".format(weight_path, self.epoch, 8, i))
                weight_cam_4.save("{}/epoch_{}_{}_{}.png".format(weight_path, self.epoch, 4, i))

                weight_cam_8r.save("{}/relative_epoch_{}_{}_{}.png".format(weight_path, self.epoch, 8, i))
                weight_cam_4r.save("{}/relative_epoch_{}_{}_{}.png".format(weight_path, self.epoch, 4, i))

                weight_cam_8 = image2tensor(weight_cam_8)
                weight_cam_4 = image2tensor(weight_cam_4)

                weight_cam_8r = image2tensor(weight_cam_8r)
                weight_cam_4r = image2tensor(weight_cam_4r)

                self.vis.img(name='weight_cam8_batch_avg_{}'.format(i), img_=weight_cam_8)
                self.vis.img(name='weight_cam4_batch_avg_{}'.format(i), img_=weight_cam_4)

                self.vis.img(name='relative_weight_cam8_batch_avg_{}'.format(i), img_=weight_cam_8r)
                self.vis.img(name='relative_weight_cam4_batch_avg_{}'.format(i), img_=weight_cam_4r)

    def __confuseMtrix(self, cm, is_vis_show=True):
        cm_img_tensor = confuseMtrix(self.env_name, cm, class_num=self.class_num)
        if is_vis_show:
            self.vis.img(name='confuseMtrix', img_=cm_img_tensor)


class PolarNetTrainProcess(TrainProcessBase):
    def model_code_block(self, data, is_training: bool):
        inputs, labels, name = data
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            for x in inputs:
                x = x.to(self.device)
            ret_inputs = inputs[0]

        else:
            inputs = inputs.to(self.device)
            ret_inputs = inputs

        labels = labels.to(self.device)

        self.optimizer.zero_grad() if is_training else ...

        outputs, weight_cam = self.model(inputs, is_training)
        torch.cuda.synchronize()

        if is_training:
            loss_ce = self.criterion(outputs, labels)
            loss_ce.backward()
            self.optimizer.step()

        return ret_inputs, outputs, labels, None, name, weight_cam
