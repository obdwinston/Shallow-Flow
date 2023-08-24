import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from scipy.interpolate import griddata
import moviepy.editor as mpy

# dam break over bump

data = 'data_bump/'
tt = 22.5
it = 100
nit = 45000

Lx = 5.6
Ly = .5
res = .005
mtd = 'nearest'

xmin = 0.
xmax = 5.6
ymin = 0.
ymax = .5
zmin = 0.
zmax = .111

figx = 12.
figy = 10.
elev = 15.
azim = -90.
roll = 0.
asx = 10.
asy = 1.
asz = 2.

# circular dam break

# data = 'data_circular/'
# tt = 5.
# it = 100
# nit = 10000

# Lx = 50.
# Ly = 50.
# res = .1
# mtd = 'linear'

# xmin = 0.
# xmax = 50.
# ymin = 0.
# ymax = 50.
# zmin = 0.
# zmax = 2.5

# figx = 12.
# figy = 10.
# elev = 15.
# azim = 0.
# roll = 0.
# asx = 1.
# asy = 1.
# asz = 1.

# program start

x = np.arange(0., Lx + res, res)
y = np.arange(0., Ly + res, res)
x, y = np.meshgrid(x, y)

cc = np.load(data + 'cc.npy')
b = np.load(data + 'b.npy')
zb = griddata(cc, b, (x, y), method=mtd)

frames = []
for nt in range(0, nit + 1, it):
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"}, figsize=(figx, figy))

    h = np.load(data + 'h{:d}.npy'.format(nt))
    zh = griddata(cc, h, (x, y), method=mtd)

    ax.plot_surface(x, y, zb, color='sienna', rstride=5, cstride=5)
    ax.plot_surface(x, y, zb + zh, color='navy',
                    rstride=5, cstride=5, alpha=.5)

    # ax.set_xlabel('x [m]')
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter('{x:.02f}')

    # ax.set_ylabel('y [m]')
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter('{x:.02f}')

    # ax.set_zlabel('z [m]')
    ax.set_zlim(zmin, zmax)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_box_aspect(aspect=(asx, asy, asz))

    fig.tight_layout()
    fig.savefig(data + 'figure{:d}.png'.format(nt))
    fig.clf()

    frames.append(data + 'figure{:d}.png'.format(nt))
    print(nt)

clip = mpy.ImageSequenceClip(frames, fps=int(len(frames)/tt))
clip.write_videofile(data + 'animation.mp4')
