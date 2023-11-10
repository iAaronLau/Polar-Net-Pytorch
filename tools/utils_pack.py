from torchvision import transforms
from PIL import Image
import glob
import torch
import cv2
import numpy as np
import os
import csv
from pathlib import Path
import torch.nn as nn
import math
import time
import random
from tools.visualizer import RedundancyVisualizer
import pandas as pd
from tools.draw_plot import AvgPlot
from Config import ConfigBase
from numpy import mean
import matplotlib
from PIL import Image, ImageDraw
from copy import deepcopy
import threading

VALID_START_POINT: int = 0


class saveThread(threading.Thread):
    def __init__(self, model_now, pth_now):
        threading.Thread.__init__(self)
        self.model_now = model_now
        self.pth_now = pth_now

    def run(self):
        torch.save(self.model_now, self.pth_now)


def int2tuple(i: int):
    return (i, i) if isinstance(i, int) else i


def str_size(input_size: tuple, input_size_b: tuple = None):
    input_size = int2tuple(input_size)
    str_input_size = "{}x{}".format(input_size[0], input_size[1])

    if input_size_b:
        input_size_b = int2tuple(input_size_b)
        str_input_size += "&{}x{}".format(input_size_b[0], input_size_b[1])

    return str_input_size


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class SaveLastModel(object):
    '''
    save the newest parameters
    '''
    def __init__(self, fold: int, save_path: str):
        self.save_path = save_path
        self.pth_last = "not_exists"
        self.fold = str(fold)
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.savethread = None

    def update(self, model_now: nn.Module, epoch_now: int):
        self.pth_now = os.path.join(self.save_path, "Model-fold-{}-state-{}.pth".format(self.fold, epoch_now))
        Path(self.pth_last).unlink(missing_ok=True)
        if isinstance(model_now, nn.DataParallel):
            model_now = model_now.module

        model_now = deepcopy(model_now.state_dict())
        self.savethread.join() if self.savethread is not None else ...
        self.savethread = saveThread(model_now, self.pth_now)
        self.savethread.start()

        self.pth_last = self.pth_now


class RecordRes(object):
    def __init__(self, fold: int, log_path: str = None):
        self.fold = fold
        if log_path:
            self.log_path = os.path.join(log_path, "csv", str(fold))
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
            self.log_path = os.path.join(log_path, "csv", str(fold), "Record_acc_auc_kappa.csv")
            Path(self.log_path).touch(exist_ok=True)
            _ = open(self.log_path, 'w').close()
            self.csv_file = open(self.log_path, "w")
            self.writer = csv.writer(self.csv_file)

    def update(self, acc_now: float, auc_now: float, kappa_now: float):
        self.writer.writerow([float(acc_now), float(auc_now), float(kappa_now)])

    def close(self):
        try:
            self.csv_file.close()
        except:
            _ = 0

    def __del__(self):
        self.close()


class RecordBestRes(object):
    def __init__(self, fold: int, log_path: str, first: str = "acc", is_save_model: bool = True):
        self.MAX_ACC = -1.0
        self.MAX_AUC = -1.0
        self.MAX_KAP = -1.0
        self.Epoch = -1
        self.first = first.lower()
        self.fold = fold
        self.is_save_model = is_save_model

        Path(log_path).mkdir(parents=True, exist_ok=True)
        self.log_path = os.path.join(log_path, "RecordBestRes.log")
        self.pth_path = os.path.join(log_path, "Model-fold-{}-First-{}.pth".format(self.fold, self.first))
        Path(self.log_path).touch(exist_ok=True)
        self.savethread = None

    def update(self, model_now: nn.Module, epoch_now: int, acc_now: float, auc_now: float, kappa_now: float):
        if self.first == "acc" or "acc" in self.first:
            if acc_now > self.MAX_ACC:
                self.__update(model_now, epoch_now, acc_now, auc_now, kappa_now)
                return

            if acc_now == self.MAX_ACC:
                if acc_now >= self.MAX_ACC:
                    self.__update(model_now, epoch_now, acc_now, auc_now, kappa_now)
                return

        if self.first == "auc" or "auc" in self.first:
            if auc_now > self.MAX_AUC:
                self.__update(model_now, epoch_now, acc_now, auc_now, kappa_now)
                return

            if auc_now == self.MAX_AUC:
                if acc_now >= self.MAX_ACC:
                    self.__update(model_now, epoch_now, acc_now, auc_now, kappa_now)
                return

        if self.first == "kappa" or "kap" in self.first:
            if kappa_now > self.MAX_KAP:
                self.__update(model_now, epoch_now, acc_now, auc_now, kappa_now)
                return

            if kappa_now == self.MAX_KAP:
                if acc_now >= self.MAX_ACC or auc_now >= self.MAX_AUC:
                    self.__update(model_now, epoch_now, acc_now, auc_now, kappa_now)
                return

    def conclusion(self):
        conclusion = "Fold {}, {} first: Best ACC= {} AUC= {} Kappa= {} @ Epoch {} .\n".format(
            self.fold, self.first, self.MAX_ACC, self.MAX_AUC, self.MAX_KAP, self.Epoch)
        print(conclusion)

        if self.log_path:
            with open(self.log_path, "a") as file:
                file.write(conclusion)
        return conclusion

    def __update(self, model_now: nn.Module, epoch_now: int, acc_now: float, auc_now: float, kap_now: float):
        if epoch_now >= VALID_START_POINT:
            self.MAX_ACC = acc_now
            self.MAX_AUC = auc_now
            self.MAX_KAP = kap_now
            self.Epoch = epoch_now
            if self.is_save_model:
                Path(self.pth_path).unlink(missing_ok=True)
                if isinstance(model_now, nn.DataParallel):
                    model_now = model_now.module

                # torch.save(model_now.state_dict(), self.pth_path)

                model_now = deepcopy(model_now.state_dict())
                self.savethread.join() if self.savethread is not None else ...
                self.savethread = saveThread(model_now, self.pth_path)
                self.savethread.start()

        return self


class SaveCSV(object):
    def __init__(self, csv_path: str, fold_num: int, epoch: int):
        csv_dir = os.path.join(csv_path, "csv", str(fold_num))
        Path(csv_dir).mkdir(parents=True, exist_ok=True)
        self.csv_path = os.path.join(csv_dir, 'Joint-fold-{}-state-{}-Result.csv'.format(fold_num, epoch))
        Path(self.csv_path).touch(exist_ok=True)
        self.csv_file = open(self.csv_path, "w")
        self.writer = csv.writer(self.csv_file)

    def write(self, data: list):
        self.writer.writerow(data)

    def close(self):
        try:
            self.csv_file.close()
        except:
            _ = 0

    def __del__(self):
        self.close()


def extract_maximum_connected_area(mat, threshold: int = 110):
    contours, _ = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(mat, [contours[k]], 0)

    _, mat = cv2.threshold(mat, threshold, 255, cv2.THRESH_BINARY)
    return mat


def tensor2array(tensor):
    array1 = tensor.cpu().detach().numpy()
    maxValue = array1.max()
    array1 = array1 * 255 / maxValue
    mat = np.uint8(array1)
    mat = mat.transpose(1, 2, 0)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    return mat


def get_concat_v(im1, im2, mod: str = "L"):
    dst = Image.new(mod, (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def os2od(img, mod: str = "L"):
    width, height = img.size
    im1 = img.crop((0, 0, width, height // 2))
    im2 = img.crop((0, height // 2, width, height))
    return get_concat_v(im2, im1, mod).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def os2od_list(images: list, mod: str = "L"):
    images = [os2od(x, mod) for x in images]
    return images


def array2image(array, mode: str = "L"):
    return Image.fromarray(array, mode=mode)


def image2array(array):
    return np.array(array)


def image2tensor(img):
    t = transforms.ToTensor()
    return t(img)


def array2tensor(array):
    return torch.tensor(array)


def arrayHStack(array_li: list):
    return np.concatenate(array_li, axis=1)


def arrayVStack(array_li: list):
    return np.concatenate(array_li, axis=0)


class FetchAllData():
    def __init__(self, mod_symbol: str, root_path: str, posi_symbol: str, nega_symbol: str, resize_to: int = 224):

        self.mod_symbol = mod_symbol
        self.root_path = root_path
        self.posi_symbol = posi_symbol
        self.nega_symbol = nega_symbol
        self.trans = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
        ])

        self.NameAndLabel = self.make_dataset()

    def make_dataset(self):
        tmp = [x for x in glob.glob(self.root_path + "/**/*.*", recursive=True) if self.mod_symbol in x and "3x3" in x]
        posi_li = [[Image.open(x).convert("L"), 1] for x in tmp if self.posi_symbol in x]
        nega_li = [[Image.open(x).convert("L"), 0] for x in tmp if self.nega_symbol in x]

        return posi_li + nega_li

    def fetch(self):
        return self.NameAndLabel


def abbr(full_list: list):
    abbr = ""
    for x in full_list:
        abbr += x[0]
    return abbr


def abbr_double(full_list: list):
    dist = "SVC-DVC-CC"
    abbr = ""
    for x in full_list[0]:
        abbr += dist[x]
    if len(full_list) == 2 and len(full_list[1]) != 0:
        abbr += "x"
        for x in full_list[1]:
            abbr += dist[x]
    return abbr


def read_txt_polar(txt_path: str):
    center_x = center_y = radius = 0
    with open(txt_path, 'r') as f:
        center_x, center_y, radius = f.readlines()
        center_x, center_y, radius = int(center_x.strip("\n")), int(center_y.strip("\n")), int(radius.strip("\n"))

    return center_x, center_y, radius


def trans2polar(img: Image, center_x: int, center_y: int, is_drop: bool = False):

    img: Image = img.copy()

    width, height = img.size
    if is_drop:
        r = int(
            min(-(center_x - height) if (center_x - 0.5 * height) >= 0 else (center_x), -(center_y - width) if
                (center_y - 0.5 * width) >= 0 else (center_y)))
    else:
        x, y = center_x, center_y
        r = int(
            math.sqrt(
                max((x * x + y * y), (x * x + (width - y) * (width - y)), ((height - x) * (height - x) + y * y),
                    ((height - x) * (height - x) + (width - y) * (width - y)))))

    circumference = int(math.pi * 2 * r)

    canvas = Image.new('L', (r, circumference), (100))
    canvas_array = np.array(canvas)
    img_array = np.array(img)

    # print(width, height, r, circumference)

    for r_ in range(r):
        for l_ in range(circumference // 1):

            theta_ = l_ / r if r_ != 0 else 0
            X = int(math.floor(r_ * math.cos(theta_))) + center_x
            Y = int(math.ceil(r_ * math.sin(theta_))) + center_y

            if X in range(height) and Y in range(width):
                pixel = img_array[X][Y]
            else:
                pixel = 150

            canvas_array[l_][r_] = pixel

    canvas = Image.fromarray(canvas_array)
    return canvas


def restore_plot_from_record(vis: RedundancyVisualizer, save_path: str, column: int = 1):
    try:
        for i in range(1, 6):
            csv_path = os.path.join(save_path, "csv", str(i), "Record_acc_auc_kappa.csv")
            data = pd.read_csv(csv_path, encoding='utf-8', header=None).values
            for x in data:
                vis.plot('Test_AUC_' + str(i), float(x[column]))
    except:
        _ = 1


def restore_all_plot_from_record(env_name_li: list, cfg: ConfigBase):
    for env_name in env_name_li:

        vis = RedundancyVisualizer(env=str(env_name).replace("save/", "").replace("./", "").replace("/", "_"),
                                   servers=cfg.vis_servers,
                                   port=cfg.vis_port)
        save_path = os.path.join("./save" if "save/" not in env_name else "./", env_name)

        restore_plot_from_record(vis=vis, save_path=save_path)
        try:
            AvgPlot(to_=cfg.stop_epoch).vis_plot(result_list=[save_path], title="AVERAGE_PLOT", vis=vis)
        except:
            print("ERROR AvgPlot:" + env_name)


class AvgMax(object):
    def __init__(self, save_path: str):
        self.precision: str = ".4f"
        self.max_acc_li = []
        self.max_auc_li = []
        self.max_kap_li = []
        self.avg_max_acc = 0
        self.avg_max_auc = 0
        self.avg_max_kap = 0
        self.save_path = save_path
        Path(save_path).mkdir(parents=True, exist_ok=True)

    def calculate(self):
        self.avg_max_acc = format(mean(self.max_acc_li), self.precision)
        self.std_max_acc = format(np.std(self.max_acc_li, ddof=1), self.precision)

        self.avg_max_auc = format(mean(self.max_auc_li), self.precision)
        self.std_max_auc = format(np.std(self.max_auc_li, ddof=1), self.precision)

        self.avg_max_kap = format(mean(self.max_kap_li), self.precision)
        self.std_max_kap = format(np.std(self.max_kap_li, ddof=1), self.precision)

        return self.avg_max_acc, self.avg_max_auc, self.avg_max_kap

    def save(self):
        self.max_acc_li += [self.avg_max_acc, self.std_max_acc]
        self.max_auc_li += [self.avg_max_auc, self.std_max_auc]
        self.max_kap_li += [self.avg_max_kap, self.std_max_kap]
        save_path = os.path.join(self.save_path, "avg_std_max.csv")
        Path(save_path).touch(exist_ok=True)
        with open(save_path, "w") as f:
            writer = csv.writer(f)
            head_line = ["name"] + [x + 1 for x in range(len(self.max_acc_li) - 2)] + ["avg", "std"]

            writer.writerow(head_line)
            writer.writerow(["max_acc"] + self.max_acc_li)
            writer.writerow(["max_auc"] + self.max_auc_li)
            writer.writerow(["max_kap"] + self.max_kap_li)

    def add(self, tp):
        try:
            max_acc = tp.record_best_acc.MAX_ACC
            max_auc = tp.record_best_auc.MAX_AUC
            max_kap = tp.record_best_kap.MAX_KAP

            self.add_value(max_acc, max_auc, max_kap)

        except:
            _ = 1
        return self

    def add_value(self, max_acc, max_auc, max_kap):
        try:
            max_acc = float(format(max_acc, self.precision))
            max_auc = float(format(max_auc, self.precision))
            max_kap = float(format(max_kap, self.precision))

            self.max_acc_li.append(max_acc)
            self.max_auc_li.append(max_auc)
            self.max_kap_li.append(max_kap)

        except:
            _ = 1
        return self

    def conclusion(self):
        avg_acc, avg_auc, _ = self.calculate()
        self.save()
        return avg_acc, avg_auc


def restore_avg_max_from_record(env_name_li: list):
    for env_name in env_name_li:
        save_path = os.path.join("./save" if "save/" not in env_name else "./", env_name)
        avg_max = AvgMax(save_path=save_path)
        # try:
        for i in range(1, 6):
            try:
                max_acc, max_auc, max_kap = 0., 0., 0.
                csv_path = os.path.join(save_path, "csv", str(i), "Record_acc_auc_kappa.csv")
                data = pd.read_csv(csv_path, encoding='utf-8', header=None).values
                for x in data[VALID_START_POINT:]:
                    acc, auc, kap = float(x[0]), float(x[1]), float(x[2])
                    max_acc = acc if acc > max_acc else max_acc
                    max_auc = auc if auc > max_auc else max_auc
                    max_kap = kap if kap > max_kap else max_kap

                avg_max.add_value(max_acc, max_auc, max_kap)
            except:
                _ = 1

                # avg_max.add_value(max_acc, max_auc, max_kap)

        avg_max.conclusion()

        # except:
        #     _ = 1


def gen_csv_report_from_record(env_name_li: list):
    save_path = os.path.join("./report", time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime()) + "_report.csv")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).touch(exist_ok=True)
    with open(save_path, "w") as f:
        writer = csv.writer(f)
        head_line = ["Index", "Environment", "ACC", "ACC_STD", "AUC", "AUC_STD", "Kappa", "Kappa_STD", "Remark"]
        writer.writerow(head_line)

        for i, env_name in enumerate(env_name_li):
            avg_max_path = os.path.join("./save" if "save/" not in env_name else "./", env_name)
            csv_path = os.path.join(avg_max_path, "avg_std_max.csv")

            data = pd.read_csv(csv_path, encoding='utf-8').values
            acc, acc_std, auc, auc_std, kpa, kpa_std = data[0][-2], data[0][-1], data[1][-2], data[1][-1], data[2][-2], data[2][
                -1]

            writer.writerow([int(i + 1), env_name] + [acc, acc_std, auc, auc_std, kpa, kpa_std])


def mode_trans(cam_weight, mode: int = 8):
    '''
    mode = 1 : IE
    mode = 4 : ETDRS
    mode = -4: hemispheric
    mode = 8 : SubETDRS
    '''
    if mode == 1:
        for i in [0, 1, 2]:
            cam_weight[:, i] = np.mean(cam_weight, axis=0)[i]
        return cam_weight

    if mode == 4:
        for line in [0, 2, 4, 6]:
            for i in [0, 1, 2]:
                l_1, l_2 = (line + 2) % 8, (line + 1) % 8
                cam_weight[l_2][i] = (cam_weight[l_2][i] + cam_weight[l_1][i]) / 2
                cam_weight[l_1][i] = cam_weight[l_2][i]
        return cam_weight

    if mode == -4:
        for line in [0, 2, 4, 6]:
            for i in [0, 1, 2]:
                l_1, l_2 = (line + 1) % 8, (line + 0) % 8
                cam_weight[l_2][i] = (cam_weight[l_2][i] + cam_weight[l_1][i]) / 2
                cam_weight[l_1][i] = cam_weight[l_2][i]
        return cam_weight

    if mode == 8:
        return cam_weight

    return cam_weight


def norm_npy(x, U=None):
    U = x[:, 1:] if U is None else U[:, 1:]

    x -= U.min()
    U -= U.min()

    x /= U.max()
    return x


def npy_cam_fusion(env, epoch_li, mode: int = 8, is_relative: bool = True, is_draw_bound: bool = False):
    npy_path_li = [[
        "{}/weight_cam/{}/epoch_{}_8_{}.npy".format(env, fold + 1, epoch, index) for fold, epoch in enumerate(epoch_li)
    ] for index in [0, 1, 2]]
    npy_li = [np.array([np.load(x) for x in npy_path_li[index]]).sum(axis=0) for index in [0, 1, 2]]

    npy_li = [norm_npy(x) for x in npy_li] if is_relative else [norm_npy(x, np.array(npy_li)) for x in npy_li]

    [
        draw_weight_cam(x, is_draw_bound=is_draw_bound, mode=mode).save("{}/m{}_r{}_i{}.png".format(env, mode, is_relative, i))
        for i, x in enumerate(npy_li)
    ]

    return npy_li


def draw_weight_cam(cam_weight, is_draw_bound: bool = False, mode: int = 8):
    line_color: str = "green"

    image = Image.new('RGB', (224, 224))
    draw_obj = ImageDraw.Draw(image)
    # colormap = matplotlib.colormaps['jet']
    colormap = matplotlib.cm.get_cmap("jet")

    cam_weight = mode_trans(cam_weight=cam_weight, mode=mode)

    cam_weight = colormap(cam_weight)
    cam_weight = np.uint8(cam_weight * 255)

    for i, line in enumerate([7, 6, 5, 4, 3, 2, 1, 0]):
        color1 = (0, 0, 0)
        color2 = tuple(cam_weight[line][1])
        color3 = tuple(cam_weight[line][2])

        draw_obj.pieslice((0, 0, 224, 224), start=-45 * (i + 1), end=-45 * i, fill=color3)
        draw_obj.pieslice((112 - 2 * 37, 112 - 2 * 37, 112 + 2 * 37, 112 + 2 * 37),
                          start=-45 * (i + 1),
                          end=-45 * i,
                          fill=color2)
        draw_obj.pieslice((112 - 0.78 * 37, 112 - 0.78 * 37, 112 + 0.78 * 37, 112 + 0.78 * 37), start=0, end=365, fill=color1)

    if is_draw_bound:
        draw_obj.ellipse((0, 0, 224, 224), outline=line_color, width=2)
        draw_obj.ellipse((112 - 2 * 37, 112 - 2 * 37, 112 + 2 * 37, 112 + 2 * 37), outline=line_color)
        draw_obj.ellipse((112 - 1 * 37, 112 - 1 * 37, 112 + 1 * 37, 112 + 1 * 37), outline=line_color)

        draw_obj.line((33, 33, 224 - 33, 224 - 33), line_color)
        draw_obj.line((224 - 33, 33, 33, 224 - 33), line_color)

        if mode == 8:
            draw_obj.line((0, 112, 224, 112), line_color)
            draw_obj.line((112, 0, 112, 224), line_color)

    return image
