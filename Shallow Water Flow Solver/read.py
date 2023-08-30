import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import moviepy.editor as mpy

# monai valley

folder = 'data_monai/'
interval = 100
tt = 11.25
nit = 22500

dry = 1e-3
h0 = .135
zmax = 0.

Lx = 5.448
Ly = 3.402
res = .01
mtd = 'linear'

# program start

x = np.arange(0., Lx + res, res)
y = np.arange(0., Ly + res, res)
xx, yy = np.meshgrid(x, y)

cc = []
b = []
lines = open(folder + 'z{:010d}.txt'.format(0)).readlines()
for line in lines:
    cc.append([float(line.split()[0]), float(line.split()[1])])
    b.append(float(line.split()[2]))
cc = np.array(cc)
b = np.array(b) - h0
zb = griddata(cc, b, (xx, yy), method=mtd)

frames = []
for it in range(0, nit, interval):

    z = []
    lines = open(folder + 'z{:010d}.txt'.format(it)).readlines()
    for line in lines:
        z.append(float(line.split()[3]))
    z = np.array(z) - h0
    zz = griddata(cc, z, (xx, yy), method=mtd)

    data = [
        go.Surface(x=x, y=y, z=zb+dry, colorscale='Earth', showscale=False),
        go.Surface(x=x, y=y, z=zz-dry, colorscale='Blues', 
                   cmin=-.02, cmax=.02, showscale=False, opacity=.8)
    ]

    fig = go.Figure(data=data)
    fig.update_layout(
        scene = {
            "xaxis": {"nticks": 20},
            "zaxis": {"nticks": 5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.5}
        },
        scene_camera_eye=dict(x=-1., y=-1.5, z=.6),
        # scene_camera_eye=dict(x=-.5, y=0., z=.1),
        width=1000, height=800,
        margin=dict(l=0., r=0., b=0., t=0.),
    )
    
    directory = folder + 'figure' + str(it) + '.png'
    fig.write_image(directory)

    frames.append(directory)
    print(it)

clip = mpy.ImageSequenceClip(frames, fps=int(len(frames)/tt))
clip.write_videofile(folder + 'animation.mp4')
