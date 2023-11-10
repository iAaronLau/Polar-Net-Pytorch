from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score, roc_auc_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from PIL import Image
from torchvision import transforms


def getAUC(y_true, y_score, num_class):
    '''AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if num_class == 2:
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def claMetrix(predictions: list, gt: list, class_num: int = 2):
    targets = torch.nn.functional.one_hot(gt, class_num)
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().detach().numpy()
    acc = np.mean(np.equal(np.argmax(predictions, 1), np.argmax(targets, 1)))
    conf = confusion_matrix(np.argmax(targets, 1), np.argmax(predictions, 1), labels=np.array(range(class_num)))

    auc = getAUC(gt.cpu().numpy(), predictions, class_num)
    PRE = precision_score(np.argmax(targets, 1), np.argmax(predictions, 1), average='weighted')
    RE = recall_score(np.argmax(targets, 1), np.argmax(predictions, 1), average='weighted')
    f1 = f1_score(np.argmax(predictions, 1), np.argmax(targets, 1), average='weighted')
    kappa = cohen_kappa_score(np.argmax(predictions, 1), np.argmax(targets, 1))

    return acc, auc, RE, PRE, f1, kappa, conf


def confuseMtrix(layer: str, cm, class_num: int = 2):
    fig_name = "./.tmp/" + str(layer) + "confuseMtrix.jpg"

    c = (np.sum(cm, axis=1)).reshape(class_num, 1)
    cm = np.around(100 * (cm / c), class_num)
    cm = 100 * (cm / np.sum(cm, axis=1))
    df_cm = pd.DataFrame(cm,
                         index=[i for i in [str(x) for x in range(class_num)]],
                         columns=[i for i in [str(x) for x in range(class_num)]])
    f, ax = plt.subplots(figsize=(5, 5))
    sn.set(font_scale=2.5)
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu", fmt="0.2f", annot_kws={"fontsize": 15})
    ax.tick_params(labelsize=30)
    f.savefig(fig_name)

    imgTransform = transforms.Compose([transforms.ToTensor()])

    return imgTransform(Image.open(fig_name).convert("RGB"))


def calc_sp_se(label, prediction):
    positive = negative = 0
    tp = tf = 0.0

    for i in range(len(label)):
        positive += label[i]

    negative = len(label) - positive

    for i in range(len(prediction)):
        if label[i] == 1 and prediction[i] == 1:
            tp += 1
        if label[i] == 0 and prediction[i] == 0:
            tf += 1

    sp = tp / positive
    se = tf / negative

    return sp, se
