import torch


def int2tuple(i: int):
    return (i, i) if isinstance(i, int) else i


def str_size(input_size: tuple, input_size_b: tuple = None):
    input_size = int2tuple(input_size)
    str_input_size = "{}x{}".format(input_size[0], input_size[1])

    if input_size_b:
        input_size_b = int2tuple(input_size_b)
        str_input_size += "&{}x{}".format(input_size_b[0], input_size_b[1])

    return str_input_size


class ConfigBase():
    token = "token for notification service"
    num_classes = 2
    pretrained = False
    is_use_cam = False
    is_save_model = True
    n_fold = [1, 2, 3, 4, 5]
    cam_save_epochs = -1
    base_lr = 2e-5
    weight_decay = 5e-5

    prior_k = None

    vis_servers = []
    vis_port = 20002

    batch_size = 28
    rate = 14
    resize_to_ori = (224, 224)
    resize_to_polar = (16 * 6 * rate, 16 * rate)
    num_epochs = 200
    stop_epoch = 199

    layers = ["SVC", "DVC", "CC"]
    in_channel = len(layers)
    layer_index_polar_ori = [[0, 1, 2], [0, 1, 2]]
    dataset_name = "Combine"
    weight = [.11, .21]
    input_size = str_size(resize_to_ori)
    patch_num_polar = (0, 0)
    patch_num_ori = (0, 0)

    def __getitem__(self, key):
        return self.__getattribute__(key)


class PolarNetConfig(ConfigBase):
    input_size = str_size(ConfigBase.resize_to_ori, ConfigBase.resize_to_polar)
    is_cam: bool = False
    use_pk: bool = False
    prior_k: torch.Tensor = None
