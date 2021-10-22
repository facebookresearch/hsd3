# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import atexit
import io
import logging
import tempfile
from base64 import b64encode
from typing import List, Optional

import imageio
import matplotlib
import matplotlib.patheffects as pe
import numpy as np
import torch as th
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from visdom import Visdom

log = logging.getLogger(__name__)


class RenderQueue:
    '''
    An asynchronous queue for plotting videos to visdom.
    '''

    def __init__(self, viz: Optional[Visdom] = None):
        self.viz = viz
        self.queue = mp.Queue()
        self.p = mp.Process(target=self.run, args=(self.queue, viz))
        self.p.start()
        self._call_close = lambda: self.close()
        atexit.register(self._call_close)

    def close(self):
        self.queue.put({'msg': 'quit'})
        self.p.join()
        atexit.unregister(self._call_close)

    def push(
        self,
        img: th.Tensor,
        s_left: List[str] = None,
        s_right: List[str] = None,
    ) -> None:
        self.queue.put(
            {'msg': 'push', 'img': img, 's_left': s_left, 's_right': s_right}
        )

    def plot(self) -> None:
        if self.viz is None:
            raise RuntimeError('No visom instance configured')
        self.queue.put({'msg': 'plot'})

    def save(self, path: str) -> None:
        self.queue.put({'msg': 'save', 'path': path})

    @staticmethod
    def run(queue: mp.Queue, viz: Optional[Visdom] = None):
        matplotlib.use('svg')

        imgs = []
        log.debug('Render queue running')
        while True:
            item = queue.get()
            msg = item['msg']
            if msg == 'quit':
                break
            elif msg == 'push':
                imgs.append(item['img'])
                if item['s_left'] or item['s_right']:
                    draw_text(
                        imgs[-1], s_left=item['s_left'], s_right=item['s_right']
                    )
            elif msg == 'plot' and viz:
                log.debug(f'Plotting video with {len(imgs)} frames to visdom')
                try:
                    plot_visdom_video(viz, imgs)
                except:
                    log.exception('Error plotting video')
                imgs.clear()
            elif msg == 'save':
                log.debug(
                    f'Saving video with {len(imgs)} frames as {item["path"]}'
                )
                try:
                    video_data = video_encode(imgs)
                    with open(item['path'], 'wb') as f:
                        f.write(video_data)
                except:
                    log.exception('Error saving video')
                imgs.clear()


def video_encode(imgs: List[th.Tensor], fps: int = 24):
    '''
    Encode a list of RGB images (HxWx3 tensors) to H264 video, return as a
    binary string.
    '''
    # TODO Can I write directly to a bytesIO object?
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        w = imageio.get_writer(
            tmp.name, format='FFMPEG', mode='I', fps=fps, codec='h264'
        )
        for img in imgs:
            w.append_data(img.numpy())
        w.close()

        data = open(tmp.name, 'rb').read()
    return data


def draw_text(
    img: th.Tensor, s_left: List[str] = None, s_right: List[str] = None
):
    '''
    Render text on top of an image (using matplotlib). Modifies the image
    in-place.
    img: The RGB image (HxWx3)
    s_left: Lines of text, left-aligned, starting from top
    s_right: Lines of text, right-aligned, starting from top
    '''
    dpi = 200
    fig = plt.figure(frameon=False)
    fig.set_size_inches(img.shape[1] / dpi, img.shape[0] / dpi)
    fig.set_dpi(dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, interpolation='none')
    fd = {'color': 'lime'}
    fs = 8
    if img.shape[0] < 400:
        fs = 6
    elif img.shape[0] < 250:
        fs = 4
    elif img.shape[0] < 150:
        fs = 2
    for i, s in enumerate(s_left if s_left is not None else []):
        if isinstance(s, tuple):
            s, c = s[0], s[1]
        else:
            c = fd
        txt = fig.text(0, 1 - i * 0.05, s, c, fontsize=fs, va='top', ha='left')
        txt.set_path_effects(
            [pe.Stroke(linewidth=0.4, foreground='black'), pe.Normal()]
        )
    for i, s in enumerate(s_right if s_right is not None else []):
        if isinstance(s, tuple):
            s, c = s[0], s[1]
        else:
            c = fd
        txt = fig.text(1, 1 - i * 0.05, s, c, fontsize=fs, va='top', ha='right')
        txt.set_path_effects(
            [pe.Stroke(linewidth=0.4, foreground='black'), pe.Normal()]
        )
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='rgba', dpi=dpi)
    buf.seek(0)
    data = np.frombuffer(buf.read(), dtype=np.uint8)
    rgba_shape = (img.shape[0], img.shape[1], 4)
    # Skip alpha channel when copying back to img
    img.copy_(th.from_numpy(data.reshape(rgba_shape)[:, :, :3].copy()))
    plt.close(fig)


def plot_visdom_video(
    viz: Visdom, images: List[th.Tensor], show_progress=False, **kwargs
):
    '''
    Plot array of RGB images as a video in Visdom.
    '''
    video_data = video_encode(images)
    encoded = b64encode(video_data).decode('utf-8')
    html = f'<video controls><source type="video/mp4" src="data:video/mp4;base64,{encoded}">Your browser does not support the video tag.</video>'
    viz.text(text=html)
