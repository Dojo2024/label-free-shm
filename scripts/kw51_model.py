"""KW51 railway bridge (Leuven) -- StaBIL model definition, translated from MATLAB.

Derived from: K. Maes, M. van de Velde, G. Lombaert (2025), "Finite element model
of railway bridge KW51", Zenodo, doi:10.5281/zenodo.17366772, licensed CC BY-NC-SA 4.0.
This translation is therefore also licensed CC BY-NC-SA 4.0 (attribution,
non-commercial, share-alike).

Line-by-line translation of data/kw51/fe_model/model_kw51.m (StaBIL 3.1) up to, but
not including, asmkm / tconstr / selectdof. Every array keeps StaBIL's layout and
MATLAB's IDs, ordering and padding. Units [kg, m, s].

    from kw51_model import build
    m = build(hb_rel=1.1, kxx=np.nan, Id_rel=0.06, Ad_rel=0.78)   # modal_analysis.m values

Returned dict (MATLAB line numbers in brackets):
  Nodes      [NodID x y z], 6079 rows incl. reference nodes (z=20) and bearing nodes 1-4 [l.230-533]
  Types      [(TypID, name)] = beam, truss, shell4 [l.551]
  Sections   [SecID A ky kz Ixx Iyy Izz]; plate/ballast rows hold thickness then NaN [l.562]
  Materials  [MatID E nu rho] [l.577]
  Elements   [EltID TypID SecID MatID n1 n2 n3 n4]; beams are [.. n1 n2 refnode NaN] [l.587-734]
  dof_extra  DOFs appended after getdof: DOF = [getdof(Elements,Types); dof_extra] [l.748-751]
  mbox       333 kg per retrofit box [l.758]
  box_masses [(dof, mass)]: MATLAB does M(I,I) += mass with [~,I] = selectdof(DOF, dof), in
             this order. Empty unless addmass==1 or diags_box is non-empty [l.760-853]
  constraints  [C1, C2, C3, C4], StaBIL Constr matrices, rows
             [Constant CoefS SlaveDOF CoefM1 MasterDOF1 CoefM2 MasterDOF2 ...] meaning
             Constant = CoefS*u_slave + sum CoefMk*u_masterk. Applied in a chain:
               [T1,~,MasterDOF1] = tconstr(C1, DOF)          l.937   ballast -> plates
               [T2,~,MasterDOF2] = tconstr(C2, MasterDOF1)   l.1231  diagonals, arch ends,
                                   plates, transverse beams -> girders / U-beams
               [T3,~,MasterDOF3] = tconstr(C3, MasterDOF2)   l.1290  girders -> bearings 1-4
               [T4,~,MasterDOF4] = tconstr(C4, MasterDOF3)   l.1295  boundary conditions
             T = T1*T2*T3*T4, Kr = T.'*K*T, Mr = T.'*M*T (l.1299-1301).
             C1, C2: 9 columns, unused trailing entries NaN. C3: 7 columns, the z rows padded
             with 0,0 (coef 0 on "DOF" 0). C4: 3 columns [0 1 dof] = dof fixed to zero.
  bc_dofs    seldof of l.1246-1287 (bearing-node DOFs fixed); C4 = [0, 1, bc_dofs] [l.1294]
  springs    [(dof, k)]: if fixed==0, Kr(I,I) += kxx with [~,I] = selectdof(MasterDOF4, dof),
             i.e. added AFTER reduction [l.1305-1310]; empty for fixed==1
  nd         16x4 diagonal node table [lower girder-side, 0.8 m above bottom, 0.8 m below top,
             arch], row = diagonal number 1-16 of track B; track A is +1000/+4000/+4000/+1000 [l.614-617]

DOF labels follow StaBIL: node + 0.01..0.06 = ux uy uz rx ry rz.
plot_elem is dropped (plotting only).
"""
import numpy as np

nan = np.nan


def _round(x):
    """MATLAB round: half away from zero (np.round is half-to-even)."""
    t = np.trunc(x)
    return t + np.sign(x) * (np.abs(x - t) >= 0.5)


def _linspace(d1, d2, n):
    """MATLAB linspace: d1 + (0:n-1).*(d2-d1)./(n-1), endpoints exact."""
    y = d1 + np.arange(n) * (d2 - d1) / (n - 1)
    y[0], y[-1] = d1, d2
    return y


def _one(a):
    """MATLAB assignment from a logical selection into a scalar/row slot: exactly one match or crash."""
    (a,) = a
    return a


def _isnum(v):
    return np.ndim(v) == 0 and np.issubdtype(np.asarray(v).dtype, np.number)


def _rot(s, m):
    """Slave rotations = master rotations (s, m: node IDs)."""
    return [[0, -1, s + d, 1, m + d, nan, nan, nan, nan] for d in (0.04, 0.05, 0.06)]


def _trans(s, m, dz):
    """Slave translations = master translations + rigid arm dz (l.910-913 pattern)."""
    return [[0, -1, s + 0.01, 1, m + 0.01, +dz, m + 0.05, nan, nan],
            [0, -1, s + 0.02, 1, m + 0.02, -dz, m + 0.04, nan, nan],
            [0, -1, s + 0.03, 1, m + 0.03, nan, nan, nan, nan]]


def build(Es_rel=1.0, hb_rel=1.0, kxx=1e13, Id_rel=1.0, Ad_rel=1.0, fixed=1, addmass=0, diags_box=()):
    # ---- robustness checks (l.157-187)
    for name, v in (('Es_rel', Es_rel), ('hb_rel', hb_rel), ('Id_rel', Id_rel), ('Ad_rel', Ad_rel)):
        assert _isnum(v) and np.isfinite(v) and v > 0, f'{name} must be a positive finite scalar.'
    assert np.ndim(fixed) == 0 and fixed in (0, 1), 'fixed must be 0 or 1.'
    assert np.ndim(addmass) == 0 and addmass in (0, 1), 'addmass must be 0 or 1.'
    if fixed == 0:
        assert _isnum(kxx) and np.isfinite(kxx) and kxx >= 0, 'kxx must be a nonnegative finite scalar when fixed==0.'
    db = np.asarray(diags_box)
    if db.size:
        assert (np.issubdtype(db.dtype, np.number) and db.size == max(db.shape + (1,))
                and np.all(np.isfinite(db))), 'diags_box must be empty or a finite numeric vector.'
        db = db.ravel().astype(float)
        grp = np.floor(db / 100)
        idx = db - grp * 100
        assert np.all(np.isin(grp, [10, 20, 30, 40])) and np.all((idx >= 1) & (idx <= 16)), \
            'diags_box entries must be 10xx/20xx/30xx/40xx with 01<=xx<=16.'
    else:
        db = np.zeros(0)

    def box(g, i):  # l.620: sum((floor(d/100)==g)+((d-floor(d/100)*100)==i)==2)>0
        return bool(np.any((np.floor(db / 100) == g) & ((db - np.floor(db / 100) * 100) == i)))

    # ---- nodes (l.189-533)
    h_ballast = 0.6 * hb_rel

    # x-coordinates [mm], left abutment -> middle (l.199-218)
    xn = [0, 658, 2313 / 3, 658 + 401, 2313 / 3 * 2, 2313]
    xn += [xn[-1] + 1878 * f for f in (1 / 2, 1)]
    xn += [xn[-1] + 3534 * f for f in (1 / 2, 1)]
    xn += [xn[-1] + 3533 * f for f in (1 / 2, 1)]
    xn += [xn[-1] + 3833 * f for f in (1 / 2, 1)]
    xn += [xn[-1] + d for d in (3834 / 2, 5900 - 3833, 3834)]
    xn += [xn[-1] + 3833 * f for f in (1 / 2, 1)]
    for _ in range(3):
        xn += [xn[-1] + 3933 * f for f in (1 / 2, 1)]
        xn += [xn[-1] + 3934 * f for f in (1 / 2, 1)]
        xn += [xn[-1] + 3933 * f for f in (1 / 2, 1)]
    xn += [115000 / 2 + 658 - 13767 - 23400, 115000 / 2 + 658 - 13767]
    xn = np.sort(np.array(xn, dtype=float))
    # middle -> right abutment (l.221-224)
    xn = np.r_[xn, xn[-1] + (xn[-1] - xn[-2::-1])]
    xn = np.r_[xn, 115000 / 2 + 658 + 3933 / 2 + 300]
    xn = np.sort(xn) / 1000
    yn = np.ones(xn.shape)
    n = len(xn)
    k = np.arange(1, n + 1)
    one = np.ones(n)

    def zarch(x):  # arch neutral axis (l.238)
        return -0.0056679878 * x ** 2 + 0.659277672 * x + 0.341695

    Nodes = np.vstack([
        [101000, 0, 0, 20],
        np.c_[101000 + k, xn, 0 * yn, 0.2719 * one],
        [102000, 0, 11.750, 20],
        np.c_[102000 + k, xn, 11.750 * yn, 0.2719 * one],
        np.c_[103000 + k[1:-1], xn[1:-1], 0 * yn[1:-1], zarch(xn[1:-1])],
        np.c_[104000 + k[1:-1], xn[1:-1], 11.750 * yn[1:-1], zarch(xn[1:-1])]])

    # diagonal connections: lower (girder) and upper (arch) x [l.249-262]
    xdl = [658 + 10600]
    xdl.append(xdl[-1] + 11500)
    for _ in range(6):
        xdl.append(xdl[-1] + 11800)
    xdl.append(xdl[-1] + 11500)
    xdl = np.array(xdl) / 1000
    xdu = [658 + 10600 + 5900]
    xdu.append(xdu[-1] + 5600 + 5900)
    for _ in range(5):
        xdu.append(xdu[-1] + 11800)
    xdu.append(xdu[-1] + 5600 + 5900)
    xdu = np.array(xdu) / 1000

    # l.265-277: exact float equality, as in MATLAB
    ndl = np.zeros(len(xdl))
    for i in range(len(xdl)):
        iSel = (np.floor(Nodes[:, 0] / 1000) == 101) & (Nodes[:, 1] == xdl[i])
        ndl[i] = _one(105000 + Nodes[iSel, 0] - np.floor(Nodes[iSel, 0] / 1000) * 1000)
    ndu = np.zeros(len(xdu))
    for i in range(len(xdu)):
        iSel = (np.floor(Nodes[:, 0] / 1000) == 103) & (Nodes[:, 1] == xdu[i])
        ndu[i] = _one(Nodes[iSel, 0])
    z773 = (80 + 693) / 1000 * np.ones(len(xdl))
    Nodes = np.vstack([Nodes, np.c_[ndl, xdl, 0 * xdl, z773]])                  # 105XXX
    Nodes = np.vstack([Nodes, np.c_[ndl + 1000, xdl, 11.750 * np.ones(len(xdl)), z773]])  # 106XXX

    # braces 107XXX-110XXX (l.287-318)
    Nebr = 6
    xbr = 0.658 + 115 / 2 + np.array([-23400 - 13767, -13767, +13767, +13767 + 23400]) / 1000
    xbr = np.tile(xbr, (Nebr + 1, 1))
    nbr = np.zeros((Nebr + 1, 4))
    ybr = np.tile(_linspace(0, 11.750, Nebr)[:, None], (1, 4))
    ybr = np.vstack([ybr[:-1], ybr[-1] - 0.85, ybr[-1]])   # extra node for damage simulation
    zbr = np.zeros((Nebr + 1, 4))
    for i in range(4):
        rx = _round(Nodes[:, 1] * 10000) == _round(xbr[0, i] * 10000)
        nbr[0, i] = _one(Nodes[(np.floor(Nodes[:, 0] / 1000) == 103) & rx, 0])
        iSel = (np.floor(Nodes[:, 0] / 1000) == 104) & rx
        nbr[-1, i] = _one(Nodes[iSel, 0])
        zbr[:, i] = _one(Nodes[iSel, 3])
        nbr[1:-1, i] = 107000 + i * 1000 + np.arange(1, Nebr)
    xbr = np.vstack([xbr, xbr[-1]])
    ybr = np.vstack([ybr, ybr[0]])
    zbr = np.vstack([zbr, 20 * np.ones(4)])
    nbr = np.vstack([nbr, np.arange(107000, 110001, 1000)])
    r = [1, 2, 3, 4, 5, 7]   # MATLAB rows [2:end-2,end], flattened column-major
    Nodes = np.vstack([Nodes, np.c_[nbr[r].ravel('F'), xbr[r].ravel('F'), ybr[r].ravel('F'), zbr[r].ravel('F')]])

    # diagonals A, 0.8 m above bottom of tube: 111XXX / 112XXX (l.324-366)
    ndlb = np.r_[ndl[:-1] + 6000, ndl[1:] + 7000]
    xdlb = np.zeros(ndlb.shape)
    zdlb = np.zeros(ndlb.shape)
    zdlb_ref = (80 + 693) / 1000
    xdlb[0] = xdl[0] + (2.383 + 0.8) / 10.9394 * 5.900
    zdlb[0] = zdlb_ref + (2.383 + 0.8) / 10.9394 * 9.212
    xdlb[1] = xdl[1] + (2.252 + 0.8) / 15.0151 * 5.900
    zdlb[1] = zdlb_ref + (2.252 + 0.8) / 15.0151 * 13.8074
    xdlb[2] = xdl[2] + (2.302 + 0.8) / 17.9606 * 5.900
    zdlb[2] = zdlb_ref + (2.302 + 0.8) / 17.9606 * 16.9639
    xdlb[3] = xdl[3] + (2.404 + 0.8) / 19.4585 * 5.900
    zdlb[3] = zdlb_ref + (2.404 + 0.8) / 19.4585 * 18.5425
    xdlb[4] = xdl[4] + (2.404 + 0.8) / 19.4585 * 5.900
    zdlb[4] = zdlb_ref + (2.404 + 0.8) / 19.4585 * 18.5425
    xdlb[5] = xdl[5] + (2.395 + 0.8) / 17.9606 * 5.900
    zdlb[5] = zdlb_ref + (2.395 + 0.8) / 17.9606 * 16.9639
    xdlb[6] = xdl[6] + (2.302 + 0.8) / 15.0151 * 5.900
    zdlb[6] = zdlb_ref + (2.302 + 0.8) / 15.0151 * 13.8074
    xdlb[7] = xdl[7] + (2.399 + 0.8) / 10.7806 * 5.600
    zdlb[7] = zdlb_ref + (2.399 + 0.8) / 10.7806 * 9.212
    xdlb[8] = xdl[1] - (2.399 + 0.8) / 10.7806 * 5.600
    zdlb[8] = zdlb_ref + (2.399 + 0.8) / 10.7806 * 9.212
    xdlb[9] = xdl[2] - (2.302 + 0.8) / 15.0151 * 5.900
    zdlb[9] = zdlb_ref + (2.302 + 0.8) / 15.0151 * 13.8074
    xdlb[10] = xdl[3] - (2.395 + 0.8) / 17.9606 * 5.900
    zdlb[10] = zdlb_ref + (2.395 + 0.8) / 17.9606 * 16.9639
    xdlb[11] = xdl[4] - (2.404 + 0.8) / 19.4585 * 5.900
    zdlb[11] = zdlb_ref + (2.404 + 0.8) / 19.4585 * 18.5425
    xdlb[12] = xdl[5] - (2.404 + 0.8) / 19.4585 * 5.900
    zdlb[12] = zdlb_ref + (2.404 + 0.8) / 19.4585 * 18.5425
    xdlb[13] = xdl[6] - (2.302 + 0.8) / 17.9606 * 5.900
    zdlb[13] = zdlb_ref + (2.302 + 0.8) / 17.9606 * 16.9639
    xdlb[14] = xdl[7] - (2.252 + 0.8) / 15.0151 * 5.900
    zdlb[14] = zdlb_ref + (2.252 + 0.8) / 15.0151 * 13.8074
    xdlb[15] = xdl[8] - (2.383 + 0.8) / 10.9394 * 5.900
    zdlb[15] = zdlb_ref + (2.383 + 0.8) / 10.9394 * 9.212
    xdlb = np.r_[xdlb, 0]
    zdlb = np.r_[zdlb, 20]
    ndlb = np.r_[ndlb, 111000]
    Nodes = np.vstack([Nodes, np.c_[ndlb, xdlb, 0 * xdlb, zdlb]])

    # diagonals A, 0.8 m below top of tube: 113XXX / 114XXX (l.372-415)
    ndub = np.r_[ndu + 10000, ndu + 11000]
    xdub = np.zeros(ndub.shape)
    zdub = np.zeros(ndub.shape)
    zdub_ref = zarch(xdu)
    zdub_ref = np.r_[zdub_ref, zdub_ref]
    xdub[0] = xdu[0] - (3.484 + 0.8) / 10.9394 * 5.900
    zdub[0] = zdub_ref[0] - (3.484 + 0.8) / 10.9394 * 9.212
    xdub[1] = xdu[1] - (2.741 + 0.8) / 15.0151 * 5.900
    zdub[1] = zdub_ref[1] - (2.741 + 0.8) / 15.0151 * 13.8074
    xdub[2] = xdu[2] - (2.559 + 0.8) / 17.9606 * 5.900
    zdub[2] = zdub_ref[2] - (2.559 + 0.8) / 17.9606 * 16.9639
    xdub[3] = xdu[3] - (2.502 + 0.8) / 19.4585 * 5.900
    zdub[3] = zdub_ref[3] - (2.502 + 0.8) / 19.4585 * 18.5425
    xdub[4] = xdu[4] - (2.4136 + 0.8) / 19.4585 * 5.900
    zdub[4] = zdub_ref[4] - (2.4136 + 0.8) / 19.4585 * 18.5425
    xdub[5] = xdu[5] - (2.402 + 0.8) / 17.9606 * 5.900
    zdub[5] = zdub_ref[5] - (2.402 + 0.8) / 17.9606 * 16.9639
    xdub[6] = xdu[6] - (1.389 + 0.8) / 15.0151 * 5.900
    zdub[6] = zdub_ref[6] - (1.389 + 0.8) / 15.0151 * 13.8074
    xdub[7] = xdu[7] - (2.652 + 0.8) / 10.7806 * 5.600
    zdub[7] = zdub_ref[7] - (2.652 + 0.8) / 10.7806 * 9.212
    xdub[15] = xdu[7] + (3.484 + 0.8) / 10.9394 * 5.900
    zdub[15] = zdub_ref[15] - (3.484 + 0.8) / 10.9394 * 9.212
    xdub[14] = xdu[6] + (2.741 + 0.8) / 15.0151 * 5.900
    zdub[14] = zdub_ref[14] - (2.741 + 0.8) / 15.0151 * 13.8074
    xdub[13] = xdu[5] + (2.559 + 0.8) / 17.9606 * 5.900
    zdub[13] = zdub_ref[13] - (2.559 + 0.8) / 17.9606 * 16.9639
    xdub[12] = xdu[4] + (2.502 + 0.8) / 19.4585 * 5.900
    zdub[12] = zdub_ref[12] - (2.502 + 0.8) / 19.4585 * 18.5425
    xdub[11] = xdu[3] + (2.4136 + 0.8) / 19.4585 * 5.900
    zdub[11] = zdub_ref[11] - (2.4136 + 0.8) / 19.4585 * 18.5425
    xdub[10] = xdu[2] + (2.402 + 0.8) / 17.9606 * 5.900
    zdub[10] = zdub_ref[10] - (2.402 + 0.8) / 17.9606 * 16.9639
    xdub[9] = xdu[1] + (1.389 + 0.8) / 15.0151 * 5.900
    zdub[9] = zdub_ref[9] - (1.389 + 0.8) / 15.0151 * 13.8074
    xdub[8] = xdu[0] + (2.652 + 0.8) / 10.7806 * 5.600
    zdub[8] = zdub_ref[8] - (2.652 + 0.8) / 10.7806 * 9.212
    xdub = np.r_[xdub, 0]
    zdub = np.r_[zdub, 20]
    ndub = np.r_[ndub, 113000]
    Nodes = np.vstack([Nodes, np.c_[ndub, xdub, 0 * xdub, zdub]])

    # diagonals C: 115XXX-118XXX (l.421-429)
    Nodes = np.vstack([Nodes, np.c_[ndlb + 4000, xdlb, 11.750 * np.ones(xdlb.shape), zdlb]])
    Nodes = np.vstack([Nodes, np.c_[ndub + 4000, xdub, 11.750 * np.ones(xdub.shape), zdub]])

    # U-shaped beams 122XXX-133XXX (l.434-439)
    yu = (11750 / 2 - 375 - 5 * 750 + np.arange(0, 11 * 750 + 1, 750)) / 1000
    for i in range(1, len(yu) + 1):
        Nodes = np.vstack([Nodes, [121000 + i * 1000, 0, yu[i - 1], 20],
                           np.c_[121000 + i * 1000 + k, xn, yu[i - 1] * yn, 1.011 * one]])

    # transverse beams 141XXX-173XXX (l.443-478)
    xtr = [0, 2313]
    for d in (1878, 3534, 3533, 3833, 3834, 3833):
        xtr.append(xtr[-1] + d)
    for _ in range(3):
        for d in (3933, 3934, 3933):
            xtr.append(xtr[-1] + d)
    xtr = np.array(xtr, dtype=float)
    xtr = np.r_[xtr, xtr[-1] + (xtr[-1] - xtr[-2::-1])]
    xtr = xtr / 1000
    xtr = np.tile(xtr, (14, 1))
    ytr = np.tile(np.r_[0, yu, 11.75][:, None], (1, 33))
    ztr = 0.2539 * np.ones(xtr.shape)
    ntr = np.zeros(xtr.shape)
    for i in range(1, 34):
        for j in range(1, 15):
            ntr[j - 1, i - 1] = 140000 + i * 1000 + j
    xtr = np.vstack([xtr, xtr[-1]])
    ytr = np.vstack([ytr, ytr[0]])
    ztr = np.vstack([ztr, 20 * np.ones(33)])
    ntr = np.vstack([ntr, np.floor(ntr[-1] / 1000) * 1000])
    Nodes = np.vstack([Nodes, np.c_[ntr.ravel('F'), xtr.ravel('F'), ytr.ravel('F'), ztr.ravel('F')]])

    # plates 181XXX-196XXX, two rows (-/+0.16 m, IDs 18i0XX / 18i5XX) per U-beam (l.483-495)
    yp = np.r_[0, 5.875 - 5.076, (11750 / 2 - 375 - 5 * 750 + np.arange(0, 11 * 750 + 1, 750)) / 1000,
               5.875 + 5.076, 11.750]
    Nodes = np.vstack([Nodes,
                       np.c_[180000 + 1000 + k, xn, yp[0] * yn, 1.156 * one],
                       np.c_[180000 + 2000 + k, xn, yp[1] * yn, 1.156 * one]])
    for i in range(3, len(yp) - 1):
        Nodes = np.vstack([Nodes,
                           np.c_[180000 + i * 1000 + k, xn, (yp[i - 1] - 0.32 / 2) * yn, 1.156 * one],
                           np.c_[180500 + i * 1000 + k, xn, (yp[i - 1] + 0.32 / 2) * yn, 1.156 * one]])
    Nodes = np.vstack([Nodes,
                       np.c_[180000 + (len(yp) - 1) * 1000 + k, xn, yp[-2] * yn, 1.156 * one],
                       np.c_[180000 + len(yp) * 1000 + k, xn, yp[-1] * yn, 1.156 * one]])

    # ballast 282XXX-295XXX (l.498-507)
    zb = (1.156 + h_ballast / 2) * one
    Nodes = np.vstack([Nodes, np.c_[280000 + 2000 + k, xn, yp[1] * yn, zb]])
    for i in range(3, len(yp) - 1):
        Nodes = np.vstack([Nodes,
                           np.c_[280000 + i * 1000 + k, xn, (yp[i - 1] - 0.32 / 2) * yn, zb],
                           np.c_[280500 + i * 1000 + k, xn, (yp[i - 1] + 0.32 / 2) * yn, zb]])
    Nodes = np.vstack([Nodes, np.c_[280000 + (len(yp) - 1) * 1000 + k, xn, yp[-2] * yn, zb]])

    # bearing nodes 1-4, 0.15 m below the girder (l.512-533)
    for nid, g, xb in ((1, 101, 1.059), (2, 102, 1.059),
                       (3, 101, (115000 + 658 - 401) / 1000), (4, 102, (115000 + 658 - 401) / 1000)):
        iSel = (np.floor(Nodes[:, 0] / 1000) == g) & (_round(Nodes[:, 1] * 10000) == _round(xb * 10000))
        Nodes = np.vstack([Nodes, np.r_[nid, _one(Nodes[iSel, 1:3]), -0.15]])

    # ---- types, sections, materials (l.551-581)
    Types = [(1, 'beam'), (2, 'truss'), (3, 'shell4')]
    Sections = np.array([
        [1, 7.688e-2, 2.93, 1.91, 1.010e-4, 1.440e-3, 1.010e-2],   # main girder
        [2, 2.097e-1, 2.04, 2.84, 6.610e-2, 3.790e-2, 5.430e-2],   # arch
        [3, 2.089e-2, 2.35, 2.30, 5.960e-4, 3.840e-4, 3.760e-4],   # diagonals
        [4, 3.289e-2, 2.00, 2.00, 5.770e-3, 2.880e-3, 2.880e-3],   # braces
        [5, 1.339e-2, 2.21, 2.68, 3.060e-4, 1.640e-4, 2.900e-4],   # U-shaped beams
        [6, 5.244e-2, 2.65, 2.08, 2.760e-5, 9.010e-4, 5.630e-3],   # transverse beams
        [7, 0.015, nan, nan, nan, nan, nan],                       # plate thickness
        [8, h_ballast, nan, nan, nan, nan, nan],                   # ballast thickness
        [9, 2.089e-2 * Ad_rel, 2.35, 2.30, 5.960e-4, 3.840e-4 * Id_rel, 3.760e-4],  # diagonal connections
        [10, 2.089e-2 * 0.94, 2.35, 2.30, 5.960e-4, 3.840e-4 * 0.085, 3.760e-4]])   # ... after retrofit
    Materials = np.array([
        [1, 200e9 * Es_rel, 0.3, 7850],   # steel beam members
        [2, 200e9 * Es_rel, 0.3, 7850],   # steel plates
        [3, 550e6, 0.3, 1700],            # porphyry ballast
        [4, 200e9 * Es_rel, 0.3, 7850]])  # diagonal connections

    # ---- elements (l.587-734); beams: [EltID 1 Sec Mat n1 n2 refnode]
    E = []
    for i in range(1, n):        # main girder A, C
        E.append([101000 + i, 1, 1, 1, 101000 + i, 101001 + i, 101000])
    for i in range(1, n):
        E.append([102000 + i, 1, 1, 1, 102000 + i, 102001 + i, 102000])
    for i in range(2, n - 1):    # arch A, C
        E.append([103000 + i, 1, 2, 1, 103000 + i, 103001 + i, 101000])
    for i in range(2, n - 1):
        E.append([104000 + i, 1, 2, 1, 104000 + i, 104001 + i, 102000])

    nd = ndl[np.r_[0:8, 1:9]]
    nd = np.c_[nd, ndlb[:-1], ndub[:-1], ndu[np.r_[0:8, 0:8]]]
    nd = nd[np.array([16, 8, 15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1]) - 1]
    ndref = ndlb[-1]
    for i in range(1, 17):       # diagonals A (track B): bottom / middle / top
        d = nd[i - 1]
        E.append([105000 + (i - 1) * 3 + 1, 1, 10 if box(10, i) else 9, 4, d[0], d[1], ndref])
        E.append([105000 + (i - 1) * 3 + 2, 1, 3, 1, d[1], d[2], ndref])
        E.append([105000 + (i - 1) * 3 + 3, 1, 10 if box(20, i) else 9, 4, d[2], d[3], ndref])
    for i in range(1, 17):       # diagonals C (track A)
        d = nd[i - 1]
        E.append([106000 + (i - 1) * 3 + 1, 1, 10 if box(30, i) else 9, 4, d[0] + 1000, d[1] + 4000, ndref + 4000])
        E.append([106000 + (i - 1) * 3 + 2, 1, 3, 1, d[1] + 4000, d[2] + 4000, ndref + 4000])
        E.append([106000 + (i - 1) * 3 + 3, 1, 10 if box(40, i) else 9, 4, d[2] + 4000, d[3] + 1000, ndref + 4000])
    for i in range(1, 5):        # braces
        for j in range(1, Nebr + 1):
            E.append([107000 + (i - 1) * 1000 + j, 1, 4, 1, nbr[j - 1, i - 1], nbr[j, i - 1], nbr[-1, i - 1]])
    for i in range(1, 13):       # U-shaped beams
        for j in range(1, n):
            b = 121000 + i * 1000
            E.append([b + j, 1, 5, 1, b + j, b + j + 1, b])
    for i in range(1, 34):       # transverse beams (EltID 140000+row*1000+col)
        for j in range(1, 14):
            E.append([140000 + j * 1000 + i, 1, 6, 1, ntr[j - 1, i - 1], ntr[j, i - 1], ntr[-1, i - 1]])
    E = np.array(E, dtype=float)
    E = np.c_[E, nan * np.ones(len(E))]   # l.693: beams get a NaN 8th column

    S = []                        # shell4: [EltID 3 Sec Mat n1 n2 n3 n4]
    for j in range(1, n):        # plates
        S.append([181000 + j, 3, 7, 2, 181000 + j, 181000 + j + 1, 182000 + j + 1, 182000 + j])
        S.append([182000 + j, 3, 7, 2, 182000 + j, 182000 + j + 1, 183000 + j + 1, 183000 + j])
    for i in range(3, len(yp) - 1):
        for j in range(1, n):
            S.append([180000 + i * 1000 + j, 3, 7, 2, 180500 + i * 1000 + j, 180500 + i * 1000 + j + 1,
                      180000 + (i + 1) * 1000 + j + 1, 180000 + (i + 1) * 1000 + j])
    for j in range(1, n):
        a, c = 180000 + (len(yp) - 1) * 1000, 180000 + len(yp) * 1000
        S.append([a + j, 3, 7, 2, a + j, a + j + 1, c + j + 1, c + j])
    for j in range(1, n):        # ballast
        S.append([282000 + j, 3, 8, 3, 282000 + j, 282000 + j + 1, 283000 + j + 1, 283000 + j])
    for i in range(3, len(yp) - 1):
        for j in range(1, n):
            S.append([280000 + i * 1000 - 500 + j, 3, 8, 3, 280000 + i * 1000 + j + 1, 280000 + i * 1000 + j,
                      280500 + i * 1000 + j, 280500 + i * 1000 + j + 1])
            S.append([280000 + i * 1000 + j, 3, 8, 3, 280500 + i * 1000 + j, 280500 + i * 1000 + j + 1,
                      280000 + (i + 1) * 1000 + j + 1, 280000 + (i + 1) * 1000 + j])
    Elements = np.vstack([E, np.array(S, dtype=float)])

    # ---- DOF extras for bearing nodes (l.751)
    dof_extra = np.r_[np.arange(1, 5) + 0.01, np.arange(1, 5) + 0.02, np.arange(1, 5) + 0.03]

    # ---- retrofit box masses (l.758-853): x, y, z of the 0.8 m nodes of each box
    mbox = 333
    sides = ((1, 0, 10), (2, 0, 20), (1, 4000, 30), (2, 4000, 40))   # (nd column, node offset, group)
    box_masses = []
    if addmass == 1 and db.size == 0:
        for c, off, g in sides:
            for i in range(nd.shape[0]):
                box_masses += [(float(nd[i, c] + off + d), mbox) for d in (0.01, 0.02, 0.03)]
    elif db.size != 0:
        for c, off, g in sides:
            for i in range(nd.shape[0]):
                if box(g, i + 1):
                    box_masses += [(float(nd[i, c] + off + d), mbox) for d in (0.01, 0.02, 0.03)]

    # ---- constraints; Nodes is final from here on
    fl = np.floor(Nodes[:, 0] / 1000)
    real = Nodes[:, 3] < 20
    rx = _round(Nodes[:, 1] * 10000)

    # C1: ballast (slave) -> plates (master) (l.883-935)
    C1 = []
    for i in range(2, len(yp)):
        for j in range(n):
            xs = rx == _round(xn[j] * 10000)
            N1 = Nodes[(fl == 180 + i) & real & xs]
            N2 = Nodes[(fl == 280 + i) & real & xs]
            dz = N2[0, 3] - N1[0, 3]
            C1 += _rot(N2[0, 0], N1[0, 0]) + _trans(N2[0, 0], N1[0, 0], dz)
            if 2 < i < len(yp) - 1:
                dz = N2[1, 3] - N1[1, 3]
                C1 += _rot(N2[1, 0], N1[1, 0]) + _trans(N2[1, 0], N1[1, 0], dz)
    C1 = np.array(C1)

    # C2 (l.941-1229); "Node(k)" in MATLAB is linear (column-major) indexing -> ravel('F')
    C2 = []
    for i in range(len(xdl)):    # diagonal bottoms (slave) -> main girders (master)
        xs = rx == _round(xdl[i] * 10000)
        for gm, gs in ((101, 105), (102, 106)):
            N1 = Nodes[(fl == gm) & xs].ravel('F')
            N2 = Nodes[(fl == gs) & xs].ravel('F')
            dz = N2[3] - N1[3]
            C2 += _trans(N2[0], N1[0], dz) + _rot(N2[0], N1[0])
    for gm, gs, xa in ((101, 103, xn[1]), (101, 103, xn[-2]), (102, 104, xn[1]), (102, 104, xn[-2])):
        xs = rx == _round(xa * 10000)   # arch begin / end (slave) -> main girders
        N1 = Nodes[(fl == gm) & xs].ravel('F')
        N2 = Nodes[(fl == gs) & xs].ravel('F')
        dz = N2[3] - N1[3]
        C2 += _trans(N2[0], N1[0], dz) + _rot(N2[0], N1[0])
    for gm, gs in ((101, 181), (102, 196)):   # edge plates (slave) -> main girders A, C
        for j in range(n):
            xs = rx == _round(xn[j] * 10000)
            N1 = Nodes[(fl == gm) & real & xs]
            N2 = Nodes[(fl == gs) & real & xs]
            dz = N2[0, 3] - N1.ravel('F')[3]
            C2 += _rot(N2[0, 0], N1[0, 0]) + _trans(N2[0, 0], N1[0, 0], dz)
    for i in range(1, len(yu) + 1):   # plates (slave, both rows) -> U-shaped beams (master)
        for j in range(n):
            xs = rx == _round(xn[j] * 10000)
            N1 = Nodes[(fl == 121 + i) & real & xs]
            N2 = Nodes[(fl == 181 + i + 1) & real & xs]
            m, L1 = N1[0, 0], N1.ravel('F')
            for r in (0, 1):
                s = N2[r, 0]
                dy = N2[r, 2] - L1[2]
                dz = N2[r, 3] - L1[3]
                C2 += _rot(s, m)
                C2 += [[0, -1, s + 0.01, 1, m + 0.01, +dz, m + 0.05, -dy, m + 0.06],
                       [0, -1, s + 0.02, 1, m + 0.02, -dz, m + 0.04, nan, nan],
                       [0, -1, s + 0.03, 1, m + 0.03, +dy, m + 0.04, nan, nan]]
    for i in range(1, ntr.shape[0]):  # transverse beams (slave) -> girders / U-beams (master)
        for j in range(1, ntr.shape[1] + 1):
            if i == 1:
                g = 101
            elif i == ntr.shape[0] - 1:
                g = 102
            else:
                g = 120 + ntr[i - 1, j - 1] - np.floor(ntr[i - 1, j - 1] / 1000) * 1000
            N1 = Nodes[(fl == g) & real & (rx == _round(xtr[i - 1, j - 1] * 10000))]
            N2 = Nodes[Nodes[:, 0] == ntr[i - 1, j - 1]]
            dz = N2[0, 3] - N1.ravel('F')[3]
            C2 += _rot(N2[0, 0], N1[0, 0]) + _trans(N2[0, 0], N1[0, 0], dz)
    C2 = np.array(C2)

    # C3: main girders (slave) -> bearing nodes 1-4 (l.1238-1288); dz_bc of N1 reused for all four
    xs1, xs3 = _round(1.059 * 10000), _round((115000 + 658 - 401) / 1000 * 10000)
    iSel1 = (fl == 101) & (rx == xs1)
    dz_bc = _one(Nodes[iSel1, 3] - Nodes[Nodes[:, 0] == 1, 3])
    C3 = []
    for g, xs, b in ((101, xs1, (1.01, 1.02, 1.03)), (102, xs1, (2.01, 2.02, 2.03)),
                     (101, xs3, (3.01, 3.02, 3.03)), (102, xs3, (4.01, 4.02, 4.03))):
        m = _one(Nodes[(fl == g) & (rx == xs), 0])
        C3 += [[0, 1, m + 0.01, -dz_bc, m + 0.05, -1, b[0]],
               [0, 1, m + 0.02, +dz_bc, m + 0.04, -1, b[1]],
               [0, 1, m + 0.03, -1, b[2], 0, 0]]
    C3 = np.array(C3)

    # C4: boundary conditions (l.1246-1294)
    seldof = list(1 + np.array([0.01, 0.02, 0.03]))
    seldof += list(2 + np.array([0.01, 0.03]))
    if fixed == 0:
        seldof += list(3 + np.array([0.02, 0.03]))
    elif fixed == 1:
        seldof += list(3 + np.array([0.01, 0.02, 0.03]))
    if fixed == 0:
        seldof += [4 + 0.03]
    elif fixed == 1:
        seldof += list(4 + np.array([0.01, 0.03]))
    bc_dofs = np.array(seldof)
    C4 = np.c_[np.zeros(len(bc_dofs)), np.ones(len(bc_dofs)), bc_dofs]

    springs = [(3.01, kxx), (4.01, kxx)] if fixed == 0 else []   # l.1305-1310

    return dict(Nodes=Nodes, Types=Types, Sections=Sections, Materials=Materials, Elements=Elements,
                dof_extra=dof_extra, mbox=mbox, box_masses=box_masses, constraints=[C1, C2, C3, C4],
                bc_dofs=bc_dofs, springs=springs, nd=nd)
