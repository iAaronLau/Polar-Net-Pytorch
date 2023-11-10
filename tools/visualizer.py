import visdom
import numpy as np
import time


# Redundancy visdom server. A relative safe way to storage the results, using many servers.
class RedundancyVisualizer():
    def __init__(self, servers: list = [""], port: int = 2112, env='default', **kwargs):
        self.vis = []
        for server in servers:
            self.vis.append(Visualizer(env=env, server=server, port=port, **kwargs))

    def img(self, name, img_, width=100, height=100, **kwargs):
        for index, vis in enumerate(self.vis):
            try:
                vis.img(name=name, img_=img_, width=width, height=height, **kwargs)
            except:
                self.vis.pop(index)
                index -= 1
        return self

    def plot(self, name, y, **kwargs):
        for index, vis in enumerate(self.vis):
            try:
                vis.plot(name=name, y=y, **kwargs)
            except:
                self.vis.pop(index)
                index -= 1
        return self


class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        Do many polt operations at one time.
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs)
        self.index[name] = x + 1

    def img(self, name, img_, width=100, height=100, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_, win=name, opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
