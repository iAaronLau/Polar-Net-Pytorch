from torchvision import transforms
from PIL import Image
import random
import os
import torch.utils.data as data
import pandas as pd
import torch
import glob


class OriDataset(data.Dataset):
    def __init__(self, data_path, csv_path, config=None, layers=None, resize_to=None, is_training=True, is_cat=True):

        self.is_training = is_training
        self.is_cat = is_cat
        self.csv_path = csv_path
        self.data_path = data_path
        layers = config.layers if layers is None else layers
        resize_to = config.resize_to_ori if resize_to is None else resize_to
        self.layers = layers
        self.resize_to = resize_to if isinstance(resize_to, tuple) else (resize_to, resize_to)

        self.NameAndLabel = self.make_dataset()
        self.imgTransform = transforms.Compose([
            transforms.Resize(self.resize_to),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_paths = self.NameAndLabel[index][0:-1]

        rotate = 10
        angel = random.randint(-rotate, rotate)

        images = [Image.open(x).convert("L").rotate(angel, expand=True)
                  for x in img_paths] if self.is_training else [Image.open(x).convert("L") for x in img_paths]

        images: list = [self.imgTransform(x) for x in images]

        label = self.NameAndLabel[index][-1]

        name = img_paths[0].split('/')[-3]

        outputs = torch.cat(tuple(images), dim=0) if self.is_cat else tuple(images)
        return outputs, int(label), name

    def __len__(self):
        return len(self.NameAndLabel)

    def make_dataset(self):
        res = []
        csv_path = os.path.join(self.csv_path, "train.csv") if self.is_training else os.path.join(self.csv_path, "test.csv")
        data = pd.read_csv(csv_path, encoding='utf-8', header=None).values
        for i in data:
            path = os.path.join(self.data_path, i[0], i[1], i[2])
            aim = [x for x in glob.glob(path + "/*.*", recursive=False)]
            aim.sort(reverse=True)
            tmp = []
            for l in self.layers:
                for t in aim:
                    if l in t:
                        tmp.append(t)
                        break
            if len(tmp) == len(self.layers):
                tmp.append(i[3])
                res.append(tmp)

        print(len(res))
        random.shuffle(res)
        ct = ad = 0
        for i in res:
            if i[-1] == 1:
                ad += 1
            else:
                ct += 1

        print(ad, ct)
        return res


class PolarNetDataset(OriDataset):
    def __init__(
        self,
        data_path,
        csv_path,
        config,
        is_training=True,
        is_cat=True,
    ):
        super().__init__(
            data_path=data_path,
            csv_path=csv_path,
            layers=config.layers,
            resize_to=config.resize_to_polar,
            is_training=is_training,
            is_cat=is_cat,
        )
        self.dataset_name = config.dataset_name

    def __getitem__(self, index):
        img_paths = self.NameAndLabel[index][0:-1]
        img_paths = [x.replace(self.dataset_name, self.dataset_name + "_Polar_Drop_Interpolation") for x in img_paths]

        v_s = random.randint(0, 9) if self.is_training else 0

        images = [Image.open(x + "_v" + str(v_s) + ".png").convert("L") for x in img_paths]

        images = [self.imgTransform(x) for x in images]

        label = self.NameAndLabel[index][-1]

        name = img_paths[0].split('/')[-3]

        outputs = torch.cat(tuple(images), dim=0) if self.is_cat else tuple(images)
        return outputs, int(label), name


class OCTA500Dataset(data.Dataset):
    def __init__(self, data_path, csv_path, config=None, layers=None, resize_to=None, is_training=True, is_cat=True):

        self.is_training = is_training
        self.is_cat = is_cat
        self.csv_path = csv_path
        self.data_path = data_path
        layers = config.layers if layers is None else layers
        resize_to = config.resize_to_ori if resize_to is None else resize_to
        self.layers = layers
        self.resize_to = resize_to if isinstance(resize_to, tuple) else (resize_to, resize_to)

        self.NameAndLabel = self.make_dataset()
        self.imgTransform = transforms.Compose([
            transforms.Resize(self.resize_to),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_paths = self.NameAndLabel[index][0:-1]

        rotate = 10
        angel = random.randint(-rotate, rotate)

        images = [Image.open(x).convert("L").rotate(angel, expand=True)
                  for x in img_paths] if self.is_training else [Image.open(x).convert("L") for x in img_paths]

        images: list = [self.imgTransform(x) for x in images]

        label = self.NameAndLabel[index][-1]

        name = img_paths[0].split('/')[-3]

        outputs = torch.cat(tuple(images), dim=0) if self.is_cat else tuple(images)
        return outputs, int(label), name

    def __len__(self):
        return len(self.NameAndLabel)

    def make_dataset(self):
        res = []
        csv_path = os.path.join(self.csv_path, "train.csv") if self.is_training else os.path.join(self.csv_path, "test.csv")
        data = pd.read_csv(csv_path, encoding='utf-8', header=None).values
        for i in data:
            path_a = os.path.join(self.data_path, "OCTA(FULL)", "{}.bmp".format(i[0]))
            path_b = os.path.join(self.data_path, "OCTA(ILM_OPL)", "{}.bmp".format(i[0]))
            path_c = os.path.join(self.data_path, "OCTA(OPL_BM)", "{}.bmp".format(i[0]))
            label = int(i[2])
            res.append([path_a, path_b, path_c, label])

        print(len(res))
        random.shuffle(res)
        ct = ad = 0
        for i in res:
            if i[-1] == 1:
                ad += 1
            else:
                ct += 1

        print(ad, ct)
        return res


class OCTA500DatasetPolarTrans(OCTA500Dataset):
    def __init__(
        self,
        data_path,
        csv_path,
        config,
        is_training=True,
        is_cat=True,
    ):
        super().__init__(
            data_path=data_path,
            csv_path=csv_path,
            layers=config.layers,
            resize_to=config.resize_to_polar,
            is_training=is_training,
            is_cat=is_cat,
        )
        self.dataset_name = config.dataset_name

    def __getitem__(self, index):
        img_paths = self.NameAndLabel[index][0:-1]
        img_paths = [x.replace(self.dataset_name, self.dataset_name + "_Polar_Drop_Interpolation") for x in img_paths]

        v_s = random.randint(0, 9) if self.is_training else 0

        images = [Image.open(x + "_v" + str(v_s) + ".png").convert("L") for x in img_paths]

        images = [self.imgTransform(x) for x in images]

        label = self.NameAndLabel[index][-1]

        name = img_paths[0].split('/')[-1]

        outputs = torch.cat(tuple(images), dim=0) if self.is_cat else tuple(images)
        return outputs, int(label), name
