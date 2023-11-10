from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import csv
import numpy as np
import torch
import glob

colors = [
    "gold",
    "coral",
    "olivedrab",
    "slateblue",
    "skyblue",
]


class AvgPlot():
    def __init__(self, from_: int = 0, to_: int = 150, step_: int = 1, mark: str = "Record_acc_auc_kappa"):
        # x 采样点,即横坐标
        self.from_ = from_
        self.to_ = to_
        self.step_ = step_
        self.mark = mark
        self.clear(self.from_, self.to_, self.step_, self.mark)

    def prepare_data(self, result_list: list, item_index: int = 1):
        self.clear(self.from_, self.to_, self.step_, self.mark)
        #颜色选择
        c = 0
        for result in result_list:
            c += 1
            # listCurrent = os.listdir(result)
            listCurrent = [x for x in glob.glob(result + "/**/Record_acc_auc_kappa.csv", recursive=True) if self.mark in x]
            y_data = np.zeros((len(self.x), len(listCurrent)))
            for i in range(len(listCurrent)):
                with open(listCurrent[i], 'r') as csvFile:
                    reader = csv.reader(csvFile)
                    allItem = []
                    for item in reader:
                        allItem.append(item[item_index])
                    allItem = np.array(allItem)
                    select = allItem[self.x]
                    y_data[:, i] = select
            name = result.split('/')[-1]
            p = PlotMeta(name=name, y_data=y_data, color=colors[c])
            self.y.append(p)

        return self

    def draw(self, result_list: list, title: str = "Title", ylabel: str = "AUC", item_index: int = 1):
        self.prepare_data(result_list=result_list, item_index=item_index)
        self.plot = DrawMeanPlot(title=title, x_=self.x, y_=self.y, xlabel="Epoch", ylabel=ylabel)
        return self.plot.getImagePlot()

    def vis_plot(self, result_list: list, title: str = "Title", ylabel: str = "AUC", item_index: int = 1, vis=None):
        self.prepare_data(result_list=result_list, item_index=item_index)
        self.plot = DrawMeanPlot(title=title, x_=self.x, y_=self.y, xlabel="Epoch", ylabel=ylabel, vis=vis)
        return self

    def save(self, path: str):
        if self.plot:
            self.plot.save(path)

    def clear(self, from_: int = 0, to_: int = 150, step_: int = 1, mark: str = "Record_acc_auc_kappa"):
        self.x = np.array([x for x in range(from_, to_, step_)])
        self.y = []
        self.mark = mark
        return self


class PlotMeta():
    '''
        线类 
        # name 线的名字，会出现在图例中
        # y_data 线的y轴数据 此类会对y_data自动计算均值和方差
        # color 从 color_name.png 中查找
        # style是线的样式
        # alpha 方差线的透明度
        # is_fill_between 是否画方差线
    '''
    def __init__(self, name: str, y_data: list, color: str, style: str = ".-", alpha: bool = 0.2, is_fill_between: bool = True):
        self.alpha = float(alpha) if alpha else 1.
        self.color = color
        self.name = name
        self.y_data = y_data
        self.is_fill_between = is_fill_between
        self.style = style
        self.calparams()

    def calparams(self):
        self.mean = [np.mean(x) for x in self.y_data]
        self.std = [np.std(x) for x in self.y_data]
        # print(self.y_data)
        # print(self.std)
        # print(self.mean)
        # print()

        self.mean = np.array(self.mean)
        self.std = np.array(self.std)


class DrawMeanPlot():
    '''
    图表类 画均值及方差 的线
    # title 图标标题
    # xlabel x轴名称
    # ylable y轴名称
    # x_: list[int] 数据的x坐标列表
    # y_: list[PlotMeta] y轴的数据, 是 PlotMeta 类的一个列表 
    # figsize 图标的大小
    # dpi 图表的分辨率

    '''
    def __init__(self, title: str, xlabel: str, ylabel: str, x_, y_, figsize: tuple = (4.5, 4), dpi: int = 200, vis=None):

        self.plot = plt
        self.fig = self.plot.figure(figsize=figsize, dpi=dpi)
        self.plot.title(str(title))
        self.plot.xlabel(str(xlabel))
        self.plot.ylabel(str(ylabel))
        # self.plot.subplot(1, 1, 1)

        # self.plot.grid(True)

        self.x_ = np.array(x_)
        self.y_ = y_
        self.vis = vis
        self.draw()

    def draw(self):
        lines = []
        labels = []
        for meta in self.y_:
            if meta.is_fill_between:
                up = meta.mean + meta.std
                down = meta.mean - meta.std
                self.plot.fill_between(x=self.x_, y1=down, y2=up, alpha=meta.alpha, color=meta.color, linewidth=0.01)

            if self.vis is not None:
                for x in meta.mean:
                    self.vis.plot('Test_AUC_AVG', float(x))
            l, = self.plot.plot(self.x_, meta.mean, meta.style, color=meta.color, label="", linewidth=1, markersize=0.1)
            lines.append(l)
            labels.append(meta.name)

        self.plot.legend(handles=lines, labels=labels, loc=2)
        return self

    # 获取PIL:Image格式的图像
    def getImagePlot(self):
        self.resolve_utf8()
        canvas = FigureCanvasAgg(self.plot.gcf())
        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        return Image.fromarray(np.uint8(img))

    def save(self, path: str = "plt.png"):
        self.plot.savefig(path)
        return self

    def show(self):
        self.plot.show()
        return self

    def resolve_utf8(self):
        self.plot.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
        self.plot.rcParams['axes.unicode_minus'] = False


# solve the issue in decoding Chinese characters
def resolve_utf8():
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
