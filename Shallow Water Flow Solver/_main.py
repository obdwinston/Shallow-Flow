import math as m
import numpy as np
import matplotlib.pyplot as plt

mesh = 'bump.su2'  # mesh file
data = 'data_bump/'  # data file
check = False  # check mesh

g = 9.81  # gravitational acceleration
dry = 1e-6  # dry depth
n = .011  # Manning coefficient
tt = 45.  # total time
dt = 1e-3  # time step
it = 100  # write interval

# check initial conditions!

# program start

lines = open(mesh, "r").readlines()

nodes = []
faces = []
cells = []

cell_faces = []
face_cells = []
cell_cells = []
node_cells = []
n_node_cells = []

Sf = []  # face area
nf = []  # face normal vector
cf = []  # face centre

Vc = []  # cell volume
cc = []  # cell centre
sc = []  # cell face sign
fc = []  # adjacent cell face index (local)
nc = []  # cell face opposite node index (global)
rfc = []  # cell centre to cell face distance
rnc = []  # cell opposite node to cell centre distance

wn = []  # cell-to-node weighting factor

# cells

n_cells = int(lines[1].split()[1])
for i in range(n_cells):
    n1 = int(lines[2 + i].split()[1])
    n2 = int(lines[2 + i].split()[2])
    n3 = int(lines[2 + i].split()[3])
    cells.append([n1, n2, n3])

# nodes

n_nodes = int(lines[2 + n_cells].split()[1])
for i in range(n_nodes):
    nx = float(lines[3 + n_cells + i].split()[0])
    ny = float(lines[3 + n_cells + i].split()[1])
    nodes.append([nx, ny])

# faces

n_inflow_faces = 0
for i in range(len(lines)):
    if lines[i].split()[1].upper() == "INFLOW":
        n_inflow_faces = int(lines[i + 1].split()[1])
        for j in range(n_inflow_faces):
            n1 = int(lines[i + j + 2].split()[1])
            n2 = int(lines[i + j + 2].split()[2])
            faces.append([n1, n2])
        break

n_outflow_faces = 0
for i in range(len(lines)):
    if lines[i].split()[1].upper() == "OUTFLOW":
        n_outflow_faces = int(lines[i + 1].split()[1])
        for j in range(n_outflow_faces):
            n1 = int(lines[i + j + 2].split()[1])
            n2 = int(lines[i + j + 2].split()[2])
            faces.append([n1, n2])
        break

n_solid_faces = 0
for i in range(len(lines)):
    if lines[i].split()[1].upper() == "SOLID":
        n_solid_faces = int(lines[i + 1].split()[1])
        for j in range(n_solid_faces):
            n1 = int(lines[i + j + 2].split()[1])
            n2 = int(lines[i + j + 2].split()[2])
            faces.append([n1, n2])
        break

for i in range(n_cells):
    n1 = cells[i][0]
    n2 = cells[i][1]
    n3 = cells[i][2]

    if [n1, n2] not in faces and [n2, n1] not in faces:
        faces.append([n1, n2])
    if [n2, n3] not in faces and [n3, n2] not in faces:
        faces.append([n2, n3])
    if [n3, n1] not in faces and [n1, n3] not in faces:
        faces.append([n3, n1])
n_faces = len(faces)

start = 0
end = n_inflow_faces
inflow_faces = list(range(start, end))

start += n_inflow_faces
end += n_outflow_faces
outflow_faces = list(range(start, end))

start += n_outflow_faces
end += n_solid_faces
solid_faces = list(range(start, end))

start += n_solid_faces
end = n_faces
interior_faces = list(range(start, end))
n_interior_faces = len(interior_faces)

boundary_faces = inflow_faces + outflow_faces + solid_faces
n_boundary_faces = len(boundary_faces)

# cell_faces

for i in range(n_cells):
    n1 = cells[i][0]
    n2 = cells[i][1]
    n3 = cells[i][2]

    cell_faces_list = []
    for j in range(n_faces):
        if [n1, n2] == faces[j] or [n2, n1] == faces[j]:
            cell_faces_list.append(j)
        if [n2, n3] == faces[j] or [n3, n2] == faces[j]:
            cell_faces_list.append(j)
        if [n3, n1] == faces[j] or [n1, n3] == faces[j]:
            cell_faces_list.append(j)
    cell_faces.append(cell_faces_list)

# face_cells

for i in range(n_faces):
    face_cells_list = []
    for j in range(n_cells):
        if i in cell_faces[j]:
            face_cells_list.append(j)
    if len(face_cells_list) == 1:  # boundary face
        face_cells.append([face_cells_list[0], face_cells_list[0]])
    if len(face_cells_list) == 2:  # interior face
        face_cells.append([face_cells_list[0], face_cells_list[1]])

# Sf, nf, cf

for i in range(n_faces):
    n1 = faces[i][0]
    n2 = faces[i][1]
    n1x = nodes[n1][0]
    n1y = nodes[n1][1]
    n2x = nodes[n2][0]
    n2y = nodes[n2][1]

    Sf.append(m.sqrt((n2x - n1x) ** 2 + (n2y - n1y) ** 2))
    nf.append([(n2y - n1y) / Sf[i], -(n2x - n1x) / Sf[i]])
    cf.append([0.5 * (n1x + n2x), 0.5 * (n1y + n2y)])

# Vc, cc, sc

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

    Vc.append(0.5 * abs((n1x * (n2y - n3y) + n2x * (n3y - n1y) + n3x * (n1y - n2y))))
    cc.append([(n1x + n2x + n3x) / 3, (n1y + n2y + n3y) / 3])

    sc_list = []
    cell_cells_list = []
    for j in range(3):
        fj = cell_faces[i][j]
        if i == face_cells[fj][0]:
            sc_list.append(1)
            cell_cells_list.append(face_cells[fj][1])
        else:
            sc_list.append(-1)
            cell_cells_list.append(face_cells[fj][0])
    sc.append(sc_list)
    cell_cells.append(cell_cells_list)

# fc, nc, rfc, rnc

for i in range(n_cells):
    n1 = cells[i][0]
    n2 = cells[i][1]
    n3 = cells[i][2]

    fc_list = []
    nc_list = []
    rfc_list = []
    rnc_list = []
    for j in range(3):
        fj = cell_faces[i][j]
        cj = cell_cells[i][j]

        f1 = cell_faces[cj][0]
        f2 = cell_faces[cj][1]
        f3 = cell_faces[cj][2]
        fcj_list = [f1, f2, f3]
        fcj = fcj_list.index(fj)

        na = faces[fj][0]
        nb = faces[fj][1]
        ncj_list = [n1, n2, n3]
        ncj_list.remove(na)
        ncj_list.remove(nb)
        ncj = ncj_list[0]
        ncx = nodes[ncj][0]
        ncy = nodes[ncj][1]

        ccx = cc[i][0]
        ccy = cc[i][1]
        cfx = cf[fj][0]
        cfy = cf[fj][1]

        fc_list.append(fcj)
        nc_list.append(ncj)
        rfc_list.append(m.sqrt((cfx - ccx) ** 2 + (cfy - ccy) ** 2))
        rnc_list.append(m.sqrt((ccx - ncx) ** 2 + (ccy - ncy) ** 2))
    fc.append(fc_list)
    nc.append(nc_list)
    rfc.append(rfc_list)
    rnc.append(rnc_list)

# wn

for i in range(n_nodes):
    nx = nodes[i][0]
    ny = nodes[i][1]

    dn_list = []
    node_cells_list = []
    n_node_cells_value = 0
    for j in range(n_cells):
        if i in cells[j]:
            ccx = cc[j][0]
            ccy = cc[j][1]

            dn_list.append(1 / m.sqrt((ccx - nx) ** 2 +
                           (ccy - ny) ** 2))  # reciprocal
            node_cells_list.append(j)
            n_node_cells_value += 1
    node_cells.append(node_cells_list)
    n_node_cells.append(n_node_cells_value)

    wn_list = []
    for j in range(n_node_cells[i]):
        wn_list.append(dn_list[j] / sum(dn_list))
    wn.append(wn_list)

# check mesh

if check:
    ci = 0  # check cell
    ni = 0  # check node

    plt.figure(figsize=(20, 10))

    for i in range(n_nodes):
        nx = nodes[i][0]
        ny = nodes[i][1]

        plt.text(nx, ny, i, horizontalalignment="center",
                 verticalalignment="center")

        if i == ni:
            plt.scatter(nx, ny, c='r')

            for j in range(n_node_cells[ni]):
                cj = node_cells[ni][j]
                ccx = cc[cj][0]
                ccy = cc[cj][1]

                plt.scatter(ccx, ccy, c='b')

    for i in range(n_faces):
        n1 = faces[i][0]
        n2 = faces[i][1]
        n1x = nodes[n1][0]
        n1y = nodes[n1][1]
        n2x = nodes[n2][0]
        n2y = nodes[n2][1]

        cfx = cf[i][0]
        cfy = cf[i][1]

        plt.text(cfx, cfy, i, horizontalalignment="center",
                 verticalalignment="center")

        if i in boundary_faces:
            plt.plot([n1x, n2x], [n1y, n2y], c="r", linewidth=2.0)
        else:
            plt.plot([n1x, n2x], [n1y, n2y], c="k", linewidth=0.5)

    for i in range(n_cells):
        ccx = cc[i][0]
        ccy = cc[i][1]

        plt.text(ccx, ccy, i, horizontalalignment="center",
                 verticalalignment="center")

        if i == ci:
            ccx = cc[ci][0]
            ccy = cc[ci][1]

            for j in range(3):
                fj = cell_faces[ci][j]
                cj = cell_cells[ci][j]

                cfx = cf[fj][0]
                cfy = cf[fj][1]
                nfx = nf[fj][0]
                nfy = nf[fj][1]
                scj = sc[ci][j]

                fcj = fc[ci][j]
                ncj = nc[ci][j]
                rfcj = rfc[ci][j]
                rncj = rnc[ci][j]
                ncx = nodes[ncj][0]
                ncy = nodes[ncj][1]

                vf = np.array([cfx - ccx, cfy - ccy]) / \
                    np.linalg.norm(np.array([cfx - ccx, cfy - ccy]))
                vn = np.array([ccx - ncx, ccy - ncy]) / \
                    np.linalg.norm(np.array([ccx - ncx, ccy - ncy]))
                vfx = vf[0]
                vfy = vf[1]
                vnx = vn[0]
                vny = vn[1]

                plt.arrow(cfx, cfy, scj*nfx, scj*nfy, color='g', head_width=.5)
                # plt.arrow(ccx, ccy, rfcj*vfx, rfcj*vfy, color='r', head_width=.5)
                # plt.arrow(ncx, ncy, rncj*vnx, rncj*vny, color='b', head_width=.5)
                plt.plot([ccx, ccx + rfcj*vfx],
                         [ccy, ccy + rfcj*vfy], color='r')
                plt.plot([ncx, ncx + rncj*vnx],
                         [ncy, ncy + rncj*vny], color='b')

                print(ncj, ci, fj, cj)
                print(fj, fcj, cell_faces[cj])

    plt.axis("equal")
    plt.grid("on")
    plt.show()


def get_reconstruction():

    ###########################
    ## reconstruction scheme ##
    ###########################

    # gradient reconstruction

    for i in range(n_nodes):
        hn[i] = 0.
        qxn[i] = 0.
        qyn[i] = 0.
        zn[i] = 0.
        for j in range(n_node_cells[i]):
            cj = node_cells[i][j]
            wnj = wn[i][j]

            hn[i] += wnj*h[cj]
            qxn[i] += wnj*qx[cj]
            qyn[i] += wnj*qy[cj]
            zn[i] += wnj*z[cj]

    def psi(a, b):  # van Albada limiter
        if a*b > 0:
            return ((a**2 + 1e-16)*b + (b**2 + 1e-16)*a)/(a**2 + b**2 + 2e-16)
        else:
            return 0.

    for i in range(n_cells):
        for j in range(3):
            fj = cell_faces[i][j]
            ncj = nc[i][j]
            rfcj = rfc[i][j]
            rncj = rnc[i][j]

            na = faces[fj][0]  # node a
            nb = faces[fj][1]  # node b

            dhf = (.5*(hn[na] + hn[nb]) - h[i])/rfcj
            dhn = (h[i] - hn[ncj])/rncj
            hLj = h[i] + rfcj*psi(dhf, dhn)

            dzf = (.5*(zn[na] + zn[nb]) - z[i])/rfcj
            dzn = (z[i] - zn[ncj])/rncj
            zLj = z[i] + rfcj*psi(dzf, dzn)

            bLj = zLj - hLj

            # adaptive reconstruction

            hmin = min(abs(bLj - b[i]), .25*h[i])

            if (hLj <= hmin) or (h[i] <= dry):
                hL[i, j] = h[i]
                qxL[i, j] = qx[i]
                qyL[i, j] = qy[i]
                zL[i, j] = z[i]
                bL[i, j] = z[i] - h[i]
            else:
                dqxf = (.5*(qxn[na] + qxn[nb]) - qx[i])/rfcj
                dqxn = (qx[i] - qxn[ncj])/rncj
                qxLj = qx[i] + rfcj*psi(dqxf, dqxn)

                dqyf = (.5*(qyn[na] + qyn[nb]) - qy[i])/rfcj
                dqyn = (qy[i] - qyn[ncj])/rncj
                qyLj = qy[i] + rfcj*psi(dqyf, dqyn)

                hL[i, j] = hLj
                qxL[i, j] = qxLj
                qyL[i, j] = qyLj
                zL[i, j] = zLj
                bL[i, j] = bLj

    # hydrostatic reconstruction

    for i in range(n_cells):
        for j in range(3):
            cj = cell_cells[i][j]
            fcj = fc[i][j]

            bLj = bL[i, j]
            bRj = bL[cj, fcj]
            bmax = max(bLj, bRj)

            zLj = zL[i, j]
            uL = qxL[i, j]/hL[i, j]
            vL = qyL[i, j]/hL[i, j]

            hLj = max(0., zLj - bmax)
            qxLj = hLj*uL
            qyLj = hLj*vL

            hL[i, j] = hLj
            qxL[i, j] = qxLj
            qyL[i, j] = qyLj


def get_flux():
    Ks[:, :] = 0.
    for i in range(n_cells):
        Vci = Vc[i]

        for j in range(3):
            cj = cell_cells[i][j]
            fj = cell_faces[i][j]
            fcj = fc[i][j]  # local index
            scj = sc[i][j]
            nfx = scj*nf[fj][0]  # signed
            nfy = scj*nf[fj][1]  # signed
            Sfj = Sf[fj]

            #######################
            ## convection scheme ##
            #######################

            hLj = hL[i, j]
            qxLj = qxL[i, j]
            qyLj = qyL[i, j]

            if hLj == 0.:  # qxLj == 0. also
                uL = 0.
                vL = 0.
                uhL = 0.
                vhL = 0.
            else:
                uL = qxLj/hLj
                vL = qyLj/hLj
                uhL = uL*nfx + vL*nfy
                vhL = -uL*nfy + vL*nfx

            if fj in solid_faces:
                # Ks[i, 0] += 0.
                Ks[i, 1] += -dt/Vci*Sfj*(g*hLj**2*nfx/2.)
                Ks[i, 2] += -dt/Vci*Sfj*(g*hLj**2*nfy/2.)
            elif fj in inflow_faces:
                print('inflow')
            else:  # outflow/interior faces

                # wave speed estimates

                hRj = hL[cj, fcj]
                qxRj = qxL[cj, fcj]
                qyRj = qyL[cj, fcj]

                if hRj == 0.:  # qxRj == 0. also
                    uR = 0.
                    vR = 0.
                    uhR = 0.
                    vhR = 0.
                else:
                    uR = qxRj/hRj
                    vR = qyRj/hRj
                    uhR = uR*nfx + vR*nfy
                    vhR = -uR*nfy + vR*nfx

                hs = (1./g)*(.5*(m.sqrt(g*hLj) + m.sqrt(g*hRj)) + .25*(uhL - uhR))**2
                uhs = .5*(uhL + uhR) + m.sqrt(g*hLj) - m.sqrt(g*hRj)

                if hLj == 0.:
                    sL = uhR - 2*m.sqrt(g*hRj)
                elif hLj > 0.:
                    sL = min(uhL - m.sqrt(g*hLj), uhs - m.sqrt(g*hs))
                else:
                    print('negative depth error')
                    exit()

                if hRj == 0.:
                    sR = uhL + 2*m.sqrt(g*hLj)
                elif hRj > 0.:
                    sR = max(uhR + m.sqrt(g*hRj), uhs + m.sqrt(g*hs))
                else:
                    print('negative depth error')
                    exit()

                ss = (sL*hRj*(uhR - sR) - sR*hLj*(uhL - sL)) / \
                    (hRj*(uhR - sR) - hLj*(uhL - sL))

                # flux estimates

                if sL >= 0.:
                    Ks[i, 0] += -dt/Vci*Sfj*(qxLj*nfx + qyLj*nfy)
                    Ks[i, 1] += -dt/Vci*Sfj * \
                        ((uL*qxLj + .5*g*hLj**2)*nfx + vL*qxLj*nfy)
                    Ks[i, 2] += -dt/Vci*Sfj * \
                        (uL*qyLj*nfx + (vL*qyLj + .5*g*hLj**2)*nfy)
                elif (sL < 0.) and (ss >= 0.):
                    QhL = np.array([hLj, qxLj*nfx + qyLj*nfy])
                    QhR = np.array([hRj, qxRj*nfx + qyRj*nfy])

                    FhL = np.array(
                        [hLj*uhL, uhL*(qxLj*nfx + qyLj*nfy) + .5*g*hLj**2])
                    FhR = np.array(
                        [hRj*uhR, uhR*(qxRj*nfx + qyRj*nfy) + .5*g*hRj**2])

                    Fs = (sR*FhL - sL*FhR + sL*sR*(QhR - QhL))/(sR - sL)
                    fs1 = Fs[0]
                    fs2 = Fs[1]

                    Ks[i, 0] += -dt/Vci*Sfj*fs1
                    Ks[i, 1] += -dt/Vci*Sfj*(fs2*nfx - vhL*fs1*nfy)
                    Ks[i, 2] += -dt/Vci*Sfj*(fs2*nfy + vhL*fs1*nfx)
                elif (ss < 0.) and (sR >= 0.):
                    QhL = np.array([hLj, qxLj*nfx + qyLj*nfy])
                    QhR = np.array([hRj, qxRj*nfx + qyRj*nfy])

                    FhL = np.array(
                        [hLj*uhL, uhL*(qxLj*nfx + qyLj*nfy) + .5*g*hLj**2])
                    FhR = np.array(
                        [hRj*uhR, uhR*(qxRj*nfx + qyRj*nfy) + .5*g*hRj**2])

                    Fs = (sR*FhL - sL*FhR + sL*sR*(QhR - QhL))/(sR - sL)
                    fs1 = Fs[0]
                    fs2 = Fs[1]

                    Ks[i, 0] += -dt/Vci*Sfj*fs1
                    Ks[i, 1] += -dt/Vci*Sfj*(fs2*nfx - vhR*fs1*nfy)
                    Ks[i, 2] += -dt/Vci*Sfj*(fs2*nfy + vhR*fs1*nfx)
                elif sR < 0.:
                    Ks[i, 0] += -dt/Vci*Sfj*(qxRj*nfx + qyRj*nfy)
                    Ks[i, 1] += -dt/Vci*Sfj * \
                        ((uR*qxRj + .5*g*hRj**2)*nfx + vR*qxRj*nfy)
                    Ks[i, 2] += -dt/Vci*Sfj * \
                        (uR*qyRj*nfx + (vR*qyRj + .5*g*hRj**2)*nfy)
                else:
                    print('HLLC flux error')
                    exit()

            #########################
            ## slope source scheme ##
            #########################

            zLj = zL[i, j]
            bLj = bL[i, j]
            bRj = bL[cj, fcj]  # bL[cj, fcj] = bL[i, j] for boundary faces
            bmax = max(bLj, bRj)
            bmin = min(bmax, zLj)

            # Ks[i, 0] += 0.
            Ks[i, 1] += dt/Vci*Sfj*(-nfx*g*(hLj + h[i])*(bmin - b[i])/2.)
            Ks[i, 2] += dt/Vci*Sfj*(-nfy*g*(hLj + h[i])*(bmin - b[i])/2.)

        h[i] += Ks[i, 0]
        qx[i] += Ks[i, 1]
        qy[i] += Ks[i, 2]


def get_friction():

    ############################
    ## friction source scheme ##
    ############################

    for i in range(n_cells):
        hi = h[i]
        qxi = qx[i]
        qyi = qy[i]
        ui = qxi/hi
        vi = qyi/hi

        Cf = g*n**2/hi**(1/3)
        qh = m.sqrt(qxi**2 + qyi**2)
        Sfx = -Cf*ui*m.sqrt(ui**2 + vi**2)
        Sfy = -Cf*vi*m.sqrt(ui**2 + vi**2)
        Sbfx = Sfx/(1. + dt*Cf/hi**2*(qh + qxi**2/qh))
        Sbfy = Sfy/(1. + dt*Cf/hi**2*(qh + qyi**2/qh))

        if qxi >= 0.:
            Sbfx = max(-qxi/dt, Sbfx)
        else:
            Sbfx = min(-qxi/dt, Sbfx)

        if qyi >= 0.:
            Sbfy = max(-qyi/dt, Sbfy)
        else:
            Sbfy = min(-qyi/dt, Sbfy)

        # h[i] += 0.
        qx[i] += dt*Sbfx
        qy[i] += dt*Sbfy


# flow variables

h = np.zeros(n_cells)
hn = np.zeros(n_nodes)
hL = np.zeros((n_cells, 3))

qx = np.zeros(n_cells)
qxn = np.zeros(n_nodes)
qxL = np.zeros((n_cells, 3))

qy = np.zeros(n_cells)
qyn = np.zeros(n_nodes)
qyL = np.zeros((n_cells, 3))

z = np.zeros(n_cells)
zn = np.zeros(n_nodes)
zL = np.zeros((n_cells, 3))

b = np.zeros(n_cells)
bL = np.zeros((n_cells, 3))

Q = np.zeros((n_cells, 3))
Qs = np.zeros((n_cells, 3))
Ks = np.zeros((n_cells, 3))

# initial conditions

#######################################
## Case Study 1 : Circular Dam Break ##
#######################################
# [X] Inflow Boundary
# [O] Outflow Boundary
# [X] Solid Boundary
# [X] Slope Source
# [X] Friction Source

# b[:] = 0.
# h[:] = .5
# qx[:] = 0.
# qy[:] = 0.

# for i in range(n_cells):
#     ccx = cc[i][0]
#     ccy = cc[i][1]

#     if m.sqrt((ccx - 25.)**2 + (ccy - 25.)**2) <= 10.:
#         h[i] = 2.

########################################
## Case Study 2 : Dam Break over Bump ##
########################################
# [X] Inflow Boundary
# [X] Outflow Boundary
# [O] Solid Boundary
# [O] Slope Source
# [O] Friction Source

b[:] = 0.
h[:] = dry
qx[:] = 0.
qy[:] = 0.

for i in range(n_cells):
    ccx = cc[i][0]
    ccy = cc[i][1]

    gradient = .065/.45

    if 4. <= ccx <= 4.45:
        b[i] = gradient*(ccx - 4.)

    if 4.45 <= ccx <= 4.9:
        b[i] = -gradient*(ccx - 4.45) + .065
        h[i] = max(dry, .02 - b[i])

    if 0. <= ccx <= 2.39:
        h[i] = .111

    if 4.9 <= ccx <= 5.6:
        h[i] = .02

# time loop

np.save(data + 'cc.npy', np.array(cc))
np.save(data + 'b.npy', b)

t = 0.
nt = 0
while t < tt:

    get_friction()

    Q[:, 0] = h[:]
    Q[:, 1] = qx[:]
    Q[:, 2] = qy[:]
    z[:] = b[:] + h[:]

    get_reconstruction()
    get_flux()

    Qs[:, 0] = h[:]  # h*
    Qs[:, 1] = qx[:]  # qx*
    Qs[:, 2] = qy[:]  # qy*
    z[:] = b[:] + h[:]  # z*

    get_reconstruction()
    get_flux()

    Q = .5*((Q + Qs) + Ks)

    h[:] = Q[:, 0]
    qx[:] = Q[:, 1]
    qy[:] = Q[:, 2]
    z[:] = b[:] + h[:]

    if nt % it == 0:
        np.save(data + 'h{:d}'.format(nt), h)

    t += dt
    nt += 1
    print(t, np.max(h), np.min(h))
