import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

mesh = 'monai.su2'  # mesh file
data = 'data_monai/'  # data folder
check = True  # check data

dt = 1e-3  # time step
# refer main.f90 time step

# program start

# create mesh

lines = open(mesh, "r").readlines()

nodes = []
faces = []
cells = []

n_cells = int(lines[1].split()[1])
for i in range(n_cells):
    n1 = int(lines[2 + i].split()[1])
    n2 = int(lines[2 + i].split()[2])
    n3 = int(lines[2 + i].split()[3])
    cells.append([n1, n2, n3])

n_nodes = int(lines[2 + n_cells].split()[1])
for i in range(n_nodes):
    nx = float(lines[3 + n_cells + i].split()[0])
    ny = float(lines[3 + n_cells + i].split()[1])
    nodes.append([nx, ny])

cc = []
for i in range(n_cells):
    n1 = cells[i][0]
    n2 = cells[i][1]
    n3 = cells[i][2]
    n1x = nodes[n1][0]
    n1y = nodes[n1][1]
    n2x = nodes[n2][0]
    n2y = nodes[n2][1]
    n3x = nodes[n3][0]
    n3y = nodes[n3][1]

    cc.append([(n1x + n2x + n3x) / 3, (n1y + n2y + n3y) / 3])
cc = np.array(cc)

# interpolate bed

iy = 244  # y-value lines
ixy = 95892  # total lines

lines = open(data + 'b_raw.txt').readlines()

x = []
y = []
b = []

for i in range(iy):
    y.append(float(lines[i].split()[1]))
y = np.array(y)

for i in range(ixy):
    if (i + 1) % iy == 0:
        x.append(float(lines[i].split()[0]))
    b.append(-float(lines[i].split()[2]))
x = np.array(x)
b = np.reshape(np.array(b), (-1, iy))

f = RectBivariateSpline(x, y, b)

b = np.zeros(n_cells)
for i in range(n_cells):
    b_value = float(f(cc[i, 0], cc[i, 1]))
    b[i] = b_value
b += -np.min(b)

b_data = open(data + 'b.txt', 'w')
for i in range(n_cells):
    b_data.write(str(b[i]))
    b_data.write('\n')
b_data.close()

# interpolate inflow height

tt = 22.5  # total time
h0 = .135  # initial height

lines = open(data + 'ht_raw.txt').readlines()

t = []
zt = []

for line in lines:
    t.append(float(line.split()[0]))
    zt.append(float(line.split()[1]))
t = np.array(t)
zt = np.array(zt)

f = interp1d(t, zt)

t_new = np.arange(0., tt + dt, dt)
ht = np.zeros(len(t_new))
ht_data = open(data + 'ht.txt', 'w')
for i in range(len(t_new)):
    ht_value = float(f(t_new[i])) + h0
    ht[i] = ht_value
    ht_data.write(str(ht_value))
    ht_data.write('\n')
ht_data.close()

# check data

Lx = 5.448  # domain x-length
Ly = 3.402  # domain y-length
res = .01  # surface resolution
mtd = 'nearest'  # interpolation method

h = np.zeros(n_cells)
for i in range(n_cells):
    h[i] = max(0., h0 - b[i])

if check:
    x = np.arange(0., Lx + res, res)
    y = np.arange(0., Ly + res, res)
    x, y = np.meshgrid(x, y)
    zb = griddata(cc, b, (x, y), method=mtd)
    zh = griddata(cc, h, (x, y), method=mtd)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, y, zb, color='sienna', rstride=5, cstride=5)
    ax.plot_surface(x, y, zb + zh, color='navy',
                    rstride=5, cstride=5, alpha=0.5)
    plt.show()

    plt.plot(t, zt, '-o', t_new, ht, '.')
    plt.show()
