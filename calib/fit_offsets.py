"""Mesure le gain des servos 1 à 4 du DOFBOT Pro et le zéro des servos 2, 3 et 4, à partir des images de
capture_robot.py.

    uv run python calib/fit_offsets.py --board          # écrit la mire à imprimer (calib/charuco_A4.pdf)
    uv run python calib/fit_offsets.py --plan           # recalcule les poses de prise de vue (calib/poses.csv)
    uv run python calib/fit_offsets.py --square 38.0    # ajuste sur calib/data ; --square = côté mesuré d'une case [mm]

La caméra du bras regarde une mire posée à plat devant lui. Pour chaque image, le modèle MuJoCo prédit où la
caméra se trouve d'après les angles relus sur les servos ; on cherche le gain et le zéro des servos, la position
de la mire et celle de la caméra sur le bras qui font coïncider les coins de la mire prédits et détectés.
Résultat dans <data>/offsets.csv. La méthode et ses limites sont décrites dans README.md.
"""
import argparse
import csv
import itertools
import json
import os
import re
import sys
from types import SimpleNamespace

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
DEG = np.pi / 180

# Mire ChArUco, sur une feuille A4 en paysage.
SQUARES = (7, 5)   # cases en largeur et en hauteur
SQUARE = 0.038     # côté d'une case [m] pour une impression à 100 %
MARKER = 0.75      # côté d'un marqueur / côté d'une case
DICTIONARY = cv2.aruco.DICT_4X4_50
MIN_CORNERS = 8    # coins de la mire à détecter pour garder une image
FINE_SIGMA = 0.05  # incertitude [deg] d'un angle relu sans l'arrondi d'Arm_Lib (un pas du servo vaut 0,08°)

JOINTS = ["Arm1_Joint", "Arm2_Joint", "Arm3_Joint", "Arm4_Joint"]  # articulations entre le socle et la caméra
FITTED = [2, 3, 4]  # servos dont on cherche le zéro
# Arm_Lib tronque l'angle relu à l'entier inférieur, puis renvoie 180 - angle pour les servos 2, 3 et 4 : la
# lecture est donc arrondie vers le bas pour le servo 1 et vers le haut pour les servos 2 à 4 (visible dans
# sysid/data : en fin d'échelon, lecture - consigne vaut 0 ou -1 pour le servo 1, 0 ou +1 pour les servos 2 à 4).
ROUND_UP = [False, True, True, True]

# Caméra : dans le repère de DaBai_DCW2_Link, la face avant du boîtier est à y = 12,7 mm et les objectifs sont
# alignés le long de x (maillage Yahboom). On suppose que l'axe optique est l'axe y de ce repère : c'est
# l'inclinaison de la caméra autour de l'axe du servo 4 qui fixe le zéro du servo 4 (voir README.md).
LENS = [0.0, 0.0127, 0.0]
CV_AXES = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # axes image (x à droite, y en bas, z en avant) dans ce repère

# Réglages de --plan. Les trois premiers viennent de la première capture sur le vrai bras (README.md).
FOCAL, IMAGE = 482.0, (640, 480)  # focale [px] et taille de l'image couleur publiées par le pilote de la caméra
LENS_X = -0.0115    # position de l'objectif couleur le long du boîtier [m]
# Hauteur de la table dans le repère du socle [m] : entre -0,013 et -0,001 selon les images retenues (le sol du
# modèle est à -0,015). On prend la plus haute, la plus prudente pour la pince.
BOARD_Z = -0.001
MIN_VIEW = 10       # coins de la mire qu'une pose prévue doit montrer
MAX_TILT = 60.0     # angle maximal [deg] entre l'axe optique et la normale à la mire
SHIFT = 0.03        # la pose doit rester bonne si la feuille est décalée d'autant [m] dans les quatre directions,
PITCH = 4.0         # ou si la caméra pointe d'autant [deg] plus haut ou plus bas que prévu
CLEARANCE = 0.015   # distance minimale [m] entre deux pièces du bras
TABLE = 0.07        # hauteur minimale [m] du bras et de la pince au-dessus de la table
RANGES = [(45, 135), (30, 150), (10, 170), (10, 170)]  # consignes explorées pour les servos 1 à 4 [deg]
HOME = [90, 90, 90, 90, 90, 120]
GRIPPER = 30        # pince ouverte pendant la capture, comme dans le modèle au repos : fermée, elle descend 2 cm plus bas
N_BASES = 2         # poses de base, autour desquelles chaque servo est balayé seul
STEP = 3            # pas des balayages [deg]
BASE_GAP = 20       # écart minimal [deg] entre deux poses de base, sur au moins un des servos 2 à 4
N_SCATTER = 24      # poses variées prises en plus des balayages


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def inverse(T):
    R = T[:3, :3].T
    return rigid(R, -R @ T[:3, 3])


def charuco(square=None):
    square = square or SQUARE
    return cv2.aruco.CharucoBoard(SQUARES, square, MARKER * square, cv2.aruco.getPredefinedDictionary(DICTIONARY))


def to_rad(deg, offsets=(0, 0, 0), gains=(1, 1, 1, 1)):
    """Angles des articulations 1 à 4 [rad] pour des angles de servo [deg] : q = gain * (deg - 90 - décalage).
    offsets = décalages des servos 2 à 4, gains = gains des servos 1 à 4."""
    return np.asarray(gains) * (np.asarray(deg, dtype=float)[..., :4] - 90 - np.append(0, offsets)) * DEG


class Chain:
    """Cinématique du modèle MuJoCo entre le socle et Arm4_Link, le segment qui porte la caméra."""

    def __init__(self, xml):
        self.model = model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(model)
        self.adr = [model.joint(j).qposadr[0] for j in JOINTS]
        # chaîne socle -> Arm4_Link : place de chaque corps dans le précédent et axe de son articulation, pour
        # calculer les poses de toutes les images à la fois sans passer par MuJoCo
        bodies = [model.jnt_bodyid[model.joint(j).id] for j in JOINTS]
        self.pos = model.body_pos[bodies].copy()
        self.rot = [Rotation.from_quat(np.roll(model.body_quat[b], -1)).as_matrix() for b in bodies]
        self.axis = np.array([model.jnt_axis[model.joint(j).id] for j in JOINTS])
        cam = model.body("DaBai_DCW2_Link")
        self.mount = rigid(Rotation.from_quat(np.roll(cam.quat, -1)).as_matrix(), cam.pos)  # d'après le modèle 3D
        q = np.array([0.3, -0.4, 0.5, 0.6])
        self.data.qpos[self.adr] = q
        mujoco.mj_kinematics(model, self.data)
        if not np.allclose(self.pose(q)[:3, 3], self.data.xpos[bodies[-1]], atol=1e-6):
            sys.exit("%s: the arm is not the plain chain base -> Arm4_Link that this script expects" % xml)

    def arm4(self, q):
        """Poses de Arm4_Link dans le repère du socle pour des angles q (n, 4) [rad] des articulations 1 à 4 :
        rotations (n, 3, 3) et positions (n, 3)."""
        q = np.atleast_2d(q)
        R, t = np.tile(np.eye(3), (len(q), 1, 1)), np.zeros((len(q), 3))
        for i in range(4):
            t = t + R @ self.pos[i]
            R = R @ self.rot[i] @ Rotation.from_rotvec(np.outer(q[:, i], self.axis[i])).as_matrix()
        return R, t

    def pose(self, q):
        """Pose (4x4) de Arm4_Link pour un seul jeu d'angles q [rad]."""
        R, t = self.arm4(q)
        return rigid(R[0], t[0])

    def camera(self, p, quarter=0):
        """Pose (4x4) du repère image dans Arm4_Link. p = roulis et lacet [deg] puis déplacement de l'objectif
        [m] par rapport au montage nominal ; quarter = nombre de quarts de tour de l'image autour de l'axe
        optique. Le tangage, c'est-à-dire la rotation autour de l'axe du servo 4, reste celui du modèle 3D."""
        R = Rotation.from_euler("z", p[1], degrees=True) * Rotation.from_euler("y", p[0] + 90 * quarter, degrees=True)
        return self.mount @ rigid(R.as_matrix() @ CV_AXES, LENS + np.asarray(p[2:5]))


# ---------------------------------------------------------------------------------------------- mire à imprimer

def write_board(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    px = 200  # pixels par case
    image = charuco().generateImage((SQUARES[0] * px, SQUARES[1] * px))
    page = np.array([297.0, 210.0])  # A4 en paysage [mm]
    size = np.array(SQUARES) * SQUARE * 1000
    fig = plt.figure(figsize=page / 25.4)
    ax = fig.add_axes([*((page - size) / 2 / page), *(size / page)])
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none")
    ax.axis("off")
    fig.text(0.5, 0.018, "Imprimer à 100 %% (taille réelle, sans ajustement à la page). Côté d'une case : %.1f mm ; "
             "le mesurer et donner la valeur à fit_offsets.py --square." % (SQUARE * 1000), ha="center", fontsize=7)
    fig.savefig(path)
    print("wrote %s: print it at 100%%, then measure one square with a ruler" % path)


# ------------------------------------------------------------------------------------------ poses de prise de vue

def layout():
    """Coins des marqueurs de la mire (m, 4, 3) et, pour chaque coin de l'échiquier, ses deux marqueurs voisins."""
    board = charuco()
    markers = np.array(board.getObjPoints(), dtype=float)
    near = [np.flatnonzero(np.linalg.norm(markers.mean(axis=1) - c, axis=1) < 0.8 * SQUARE)
            for c in board.getChessboardCorners()]
    return markers, np.array(near)


def corners_seen(T, markers, near):
    """Nombre de coins de la mire détectables par une caméra dans laquelle la mire a la pose T (4x4) : ceux dont
    les deux marqueurs voisins sont entiers dans l'image. Vérifié sur la première capture du vrai bras, où ce
    compte tombe à un ou deux coins près sur le nombre de coins réellement détectés."""
    p = markers.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    if p[:, 2].min() < 0.05 or abs(T[2, 2]) < np.cos(MAX_TILT * DEG):
        return 0
    uv = FOCAL * p[:, :2] / p[:, 2:] + np.array(IMAGE) / 2
    whole = ((uv > 4) & (uv < np.array(IMAGE) - 4)).all(axis=1).reshape(-1, 4).all(axis=1)
    return int(whole[near].all(axis=1).sum())


def near_pairs(chain, deg):
    """Paires de corps à moins de leur marge l'un de l'autre pour les consignes deg des servos 1 à 4."""
    chain.data.qpos[:] = chain.model.qpos0
    chain.data.qpos[chain.adr] = to_rad(deg)
    mujoco.mj_forward(chain.model, chain.data)
    body = chain.model.geom_bodyid
    return {(body[c.geom1], body[c.geom2]) for c in chain.data.contact[:chain.data.ncon]}


def path_clear(chain, a, b, allowed):
    """Le trajet en ligne droite (dans l'espace des servos) de a à b reste-t-il dégagé ?"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return all(near_pairs(chain, a + s * (b - a)) <= allowed for s in np.linspace(0, 1, 9)[1:])


def plan(chain, centre):
    """Poses de prise de vue pour une mire centrée en `centre` (x, y dans le repère du socle) : autour de N_BASES
    poses de base, chaque servo balayé seul aussi loin que la mire reste en vue, puis N_SCATTER poses variées.
    Renvoie des couples (nom, consignes entières des servos 1 à 4), dans un ordre où chaque trajet est dégagé."""
    model, floor = chain.model, chain.model.geom("floor").id
    model.geom_pos[floor, 2] = BOARD_Z
    # un contact est signalé dès que deux pièces sont à moins de la somme de leurs marges : CLEARANCE entre deux
    # pièces du bras, TABLE entre le bras et la table
    model.geom_margin[:] = CLEARANCE / 2
    model.geom_margin[floor] = TABLE - CLEARANCE / 2
    home = HOME[:4]
    allowed = near_pairs(chain, home)  # pièces voisines par construction (caméra et poignet, par exemple)
    markers, near = layout()
    flat = np.diag([1.0, -1.0, -1.0])  # mire à plat, face imprimée vers le haut
    middle = np.append(np.array(SQUARES) * SQUARE / 2, 0)
    boards = [rigid(flat, np.array([centre[0] + dx, centre[1] + dy, BOARD_Z]) - flat @ middle)
              for dx, dy in [(0, 0), (SHIFT, 0), (-SHIFT, 0), (0, SHIFT), (0, -SHIFT)]]
    cam = chain.camera([0, 0, LENS_X, 0, 0])
    known = {}

    def valid(pose):
        """La mire est-elle bien en vue, feuille et caméra un peu déplacées comprises, et le bras dégagé ?"""
        pose = tuple(int(v) for v in pose)
        if pose not in known:
            views = [(pose, b) for b in boards] + [(pose[:3] + (pose[3] + d,), boards[0]) for d in (-PITCH, PITCH)]
            known[pose] = (all(corners_seen(inverse(chain.pose(to_rad(p)) @ cam) @ b, markers, near) >= MIN_VIEW
                               for p, b in views) and near_pairs(chain, pose) <= allowed)
        return known[pose]

    def span(pose, j):
        """Plus grande plage du servo j+1 autour de la pose, les autres servos ne bougeant pas."""
        lo = hi = pose[j]
        while lo - 1 >= RANGES[j][0] and valid(pose[:j] + (lo - 1,) + pose[j + 1:]):
            lo -= 1
        while hi + 1 <= RANGES[j][1] and valid(pose[:j] + (hi + 1,) + pose[j + 1:]):
            hi += 1
        return lo, hi

    # poses de base : celles dont le plus court des quatre balayages est le plus long
    grid = itertools.product([90], range(36, 127, 6), range(12, 121, 6), range(12, 121, 6))
    spans = {pose: [span(pose, j) for j in range(4)] for pose in grid if valid(pose)}
    bases = []
    for pose in sorted(spans, key=lambda p: -min(hi - lo for lo, hi in spans[p])):
        apart = all(np.abs(np.subtract(pose, b)[1:]).max() >= BASE_GAP for b in bases)
        if apart and path_clear(chain, bases[-1] if bases else home, pose, allowed):
            bases.append(pose)
        if len(bases) == N_BASES:
            break
    if len(bases) < N_BASES:
        sys.exit("only %d base poses see the board at %s: move it (--board-at)" % (len(bases), list(centre)))

    rows = []
    for letter, base in zip("abcdef", bases):
        rows.append((letter + "0_0", base))
        for j, (lo, hi) in enumerate(spans[base]):
            for k, v in enumerate(range(lo + (hi - lo) % STEP // 2, hi + 1, STEP)):
                rows.append(("%s%d_%d" % (letter, j + 1, k), base[:j] + (v,) + base[j + 1:]))
            rows.append(("%s0_%d" % (letter, j + 1), base))  # retour à la pose de base : dit si le bras y revient bien
        print("base pose %s: servos 1 to 4 swept over %s deg" % (list(base), ", ".join("%d-%d" % s for s in spans[base])))

    # poses variées, les plus éloignées possible les unes des autres et des poses de base
    rng = np.random.default_rng(0)
    samples = np.column_stack([rng.integers(lo, hi + 1, 40000) for lo, hi in RANGES])
    pool = np.array([s for s in samples if valid(s)], dtype=float)
    gap = np.min([np.linalg.norm(pool - b, axis=1) for b in bases], axis=0)
    chosen = []
    while len(chosen) < min(N_SCATTER, len(pool)):
        chosen.append(pool[int(np.argmax(gap))])
        gap = np.minimum(gap, np.linalg.norm(pool - chosen[-1], axis=1))
    # ordre : toujours la pose la plus proche qu'on peut rejoindre sans rien frôler
    at, order = np.array(bases[-1], dtype=float), []
    while chosen:
        reachable = [p for p in sorted(chosen, key=lambda p: np.linalg.norm(p - at)) if path_clear(chain, at, p, allowed)]
        if not reachable:
            break
        at = reachable[0]
        order.append(at)
        chosen = [p for p in chosen if p is not at]
    while order and not path_clear(chain, order[-1], home, allowed):
        order.pop()
    if not order and not path_clear(chain, bases[-1], home, allowed):
        sys.exit("no clear way back from the last base pose: change --board-at")
    return rows + [("s%d" % k, tuple(int(v) for v in p)) for k, p in enumerate(order)]


def write_poses(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([["pose"] + ["s%d" % i for i in range(1, 7)]]
                                + [[name, *pose, HOME[4], GRIPPER] for name, pose in rows])
    poses = np.array([pose for _, pose in rows])
    spans = ", ".join("%d-%d" % (lo, hi) for lo, hi in zip(poses.min(axis=0), poses.max(axis=0)))
    print("wrote %s: %d poses, servos 1 to 4 span %s deg" % (path, len(rows), spans))


# ---------------------------------------------------------------------------------------------------- ajustement

def servo_angle(lo, hi, up):
    """Angle réel d'un servo [deg] et son incertitude, d'après le plus petit et le plus grand angle relus.
    Une lecture entière ne donne qu'un intervalle d'un degré, dont on prend le milieu ; si les lectures hésitent
    entre deux valeurs, l'angle est à la frontière des deux intervalles et il est connu bien plus finement."""
    a, b = (hi - 1, lo) if up else (hi, lo + 1)
    return (a + b) / 2, max(b - a, 0.2) / 12 ** 0.5


def load_images(folder, board):
    """Lit capture.csv et détecte la mire dans chaque image. Renvoie (observations, taille de l'image)."""
    detector = cv2.aruco.CharucoDetector(board)
    corners3d = board.getChessboardCorners()
    obs, size = [], None
    with open(os.path.join(folder, "capture.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        reads = [[row["%s%d" % (k, i)] for k in ("lo", "hi")] for i in range(1, 5)]
        image = cv2.imread(os.path.join(folder, row["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None or any("" in r for r in reads):
            print("  %s: %s, skipped" % (row["image"], "unreadable" if image is None else "a servo did not answer"))
            continue
        size = image.shape[::-1]
        corners, ids, _, _ = detector.detectBoard(image)
        if ids is None or len(ids) < MIN_CORNERS:
            print("  %s: %d board corners found, skipped" % (row["image"], 0 if ids is None else len(ids)))
            continue
        deg, sigma = np.array([servo_angle(*map(float, r), up) for r, up in zip(reads, ROUND_UP)]).T
        fine = [row.get("fine%d" % i) for i in range(1, 5)]
        if all(fine):  # lectures sans l'arrondi d'Arm_Lib
            deg, sigma = np.array(fine, dtype=float), np.full(4, FINE_SIGMA)
        obs.append(SimpleNamespace(name=row["image"], pose=row["pose"], deg=deg, sigma=sigma,
                                   obj=corners3d[ids.ravel()].astype(float), img=corners.reshape(-1, 2).astype(float)))
    return obs, size


def load_intrinsics(folder, obs, size):
    """Matrice de la caméra et distorsion : celles publiées par le pilote de la caméra si capture_robot.py les a
    enregistrées, sinon une calibration sur les images de la mire elles-mêmes."""
    path = os.path.join(folder, "intrinsics.json")
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)
        K, dist = np.array(info["K"], dtype=float).reshape(3, 3), np.array(info["D"], dtype=float)
        if K[0, 0] > 0 and (info["width"], info["height"]) == tuple(size):
            print("camera: intrinsics from %s (f = %.0f px)" % (path, K[0, 0]))
            return K, dist
        print("camera: %s does not match the images, ignored" % path)
    rms, K, dist, _, _ = cv2.calibrateCamera([o.obj.astype(np.float32) for o in obs],
                                             [o.img.astype(np.float32) for o in obs], size, None, None,
                                             flags=cv2.CALIB_FIX_K3)
    print("camera: calibrated on these images (f = %.0f px, centre %.0f %.0f, rms %.2f px)"
          % (K[0, 0], K[0, 2], K[1, 2], rms))
    print("warning: these views are too alike to calibrate a camera well, expect offsets wrong by up to 1 deg; "
          "capture through ROS to get the camera's own calibration (intrinsics.json)")
    return K, dist.ravel()


def prepare(obs, K, dist):
    """Ajoute à chaque observation ses points sans distorsion (coordonnées normalisées) et la pose de la mire
    vue par la caméra, estimée sur cette image seule."""
    for o in obs:
        o.norm = cv2.undistortPoints(o.img.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        _, rvec, tvec = cv2.solvePnP(o.obj, o.img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        o.board = rigid(Rotation.from_rotvec(rvec.ravel()).as_matrix(), tvec.ravel())


def fine(obs):
    """Les angles de toutes les images ont-ils été relus sans l'arrondi d'Arm_Lib ?"""
    return all((o.sigma == FINE_SIGMA).all() for o in obs)


def rotation_gains(obs):
    """Gain des servos 1 à 4 vu par la caméra seule, sans modèle du bras : l'orientation de la caméra dans chaque
    image, donnée par la mire, comparée aux angles relus. Renvoie (gains, incertitudes), nan pour un servo qui
    n'a pas assez tourné. Le servo 1 fait tourner la caméra autour de la verticale, les servos 2 à 4 la font
    basculer vers l'avant ou l'arrière."""
    R = np.array([o.board[:3, :3].T for o in obs])  # axes de la caméra dans le repère de la mire
    # l'axe de l'image qui reste parallèle à la mire est l'axe transversal du bras, celui des servos 2 à 4
    side = R[:, :, 0 if np.abs(R[:, 2, 0]).mean() < np.abs(R[:, 2, 1]).mean() else 1].copy()
    side[:, 2] = 0
    side /= np.linalg.norm(side, axis=1, keepdims=True)
    yaw = np.degrees(np.unwrap(np.arctan2(side[:, 1], side[:, 0])))
    forward = np.cross([0, 0, 1.0], side)
    pitch = np.degrees(np.arctan2((R[:, :, 2] * forward).sum(axis=1), R[:, 2, 2]))
    deg = np.array([o.deg for o in obs])
    gains, errors = np.full(4, np.nan), np.full(4, np.nan)
    for angle, servos in ((yaw, [0]), (pitch, [1, 2, 3])):
        servos = [j for j in servos if np.ptp(deg[:, j]) >= 10]  # un servo resté en place ne dit rien de son gain
        A = np.column_stack([deg[:, servos], np.ones(len(obs))])
        if not servos or len(obs) < A.shape[1] + 3:
            continue
        slope = np.linalg.lstsq(A, angle, rcond=None)[0]
        left = angle - A @ slope
        cov = left @ left / (len(obs) - A.shape[1]) * np.linalg.inv(A.T @ A)
        gains[servos], errors[servos] = np.abs(slope[:-1]), np.sqrt(np.diag(cov)[:-1])
    return gains, errors


# Paramètres ajustés : décalages des servos 2 à 4 [deg], pose de la mire dans le repère du socle (vecteur de
# rotation, position [m]), les cinq paramètres de Chain.camera, les gains des servos 1 à 4, puis pour chaque image
# une correction [deg] de l'angle des servos 1 à 4. Cette correction est indispensable : une lecture entière se
# trompe jusqu'à 0,5°, ce qui déplace la mire de plusieurs pixels dans l'image, bien plus que l'erreur de
# détection de ses coins.
N_GLOBAL = 18
PIXEL_NOISE = 0.5   # écart-type supposé de la position d'un coin détecté [px]
GAIN_PRIOR = 0.05   # un gain s'écarte de 1 de quelques pour cent au plus
FLAT_PRIOR = 0.5    # la mire et le socle sont posés sur la même table : parallèles à 0,5° près


def stack(obs, focal):
    """Regroupe les observations en tableaux, pour évaluer le modèle sur toutes les images à la fois."""
    return SimpleNamespace(n=len(obs), focal=focal, image=np.repeat(np.arange(len(obs)), [len(o.obj) for o in obs]),
                           obj=np.concatenate([o.obj for o in obs]), norm=np.concatenate([o.norm for o in obs]),
                           deg=np.array([o.deg for o in obs]), sigma=np.array([o.sigma for o in obs]))


def residuals(x, s, chain, quarter):
    """Écarts, chacun divisé par son incertitude : entre les coins de la mire prédits et détectés dans chaque
    image, entre l'angle retenu pour chaque servo et sa lecture, puis ceux des deux a priori ci-dessus."""
    Rb = Rotation.from_rotvec(x[3:6]).as_matrix()
    cam = chain.camera(x[9:14], quarter)
    fixes = x[N_GLOBAL:].reshape(s.n, 4)
    R, t = chain.arm4(to_rad(s.deg + fixes, x[:3], x[14:18]))
    R, t = R @ cam[:3, :3], t + R @ cam[:3, 3]
    p = np.einsum("mji,mj->mi", R[s.image], s.obj @ Rb.T + x[6:9] - t[s.image])
    corners = s.focal / PIXEL_NOISE * (p[:, :2] / np.maximum(p[:, 2:], 1e-3) - s.norm)
    priors = np.append((x[14:18] - 1) / GAIN_PRIOR, np.degrees(Rb[:2, 2]) / FLAT_PRIOR)
    return np.concatenate([corners.ravel(), (fixes / s.sigma).ravel(), priors])


def solve(x0, free, s, chain, quarter):
    """Ajuste les paramètres d'indices `free` en partant de x0 ; renvoie (paramètres, résultat de least_squares)."""
    free = np.asarray(free)
    shared = free[free < N_GLOBAL]
    corners = 2 * len(s.obj)

    def full(z):
        x = x0.copy()
        x[free] = z
        return x

    def jac(z):
        x = full(z)
        r0 = residuals(x, s, chain, quarter)
        J = np.zeros((len(r0), len(free)))
        for col, k in enumerate(shared):
            y = x.copy()
            y[k] += 1e-6
            J[:, col] = (residuals(y, s, chain, quarter) - r0) / 1e-6
        if len(free) > len(shared):
            # la correction d'une image ne touche que cette image : on les déplace toutes ensemble, servo par servo
            for j in range(4):
                y = x.copy()
                y[N_GLOBAL + j::4] += 1e-5
                d = (residuals(y, s, chain, quarter) - r0) / 1e-5
                J[np.arange(corners), np.repeat(len(shared) + 4 * s.image + j, 2)] = d[:corners]
                own = 4 * np.arange(s.n) + j
                J[corners + own, len(shared) + own] = d[corners + own]
        return J

    sol = least_squares(lambda z: residuals(full(z), s, chain, quarter), x0[free], jac=jac, x_scale="jac",
                        loss="soft_l1", f_scale=3.0, max_nfev=200)
    return full(sol.x), sol


def start(obs, chain, quarter, gains):
    """Point de départ : décalages et corrections nuls, caméra nominale, et la pose de la mire donnée par l'image
    la plus cohérente avec les autres."""
    cam = chain.camera(np.zeros(5), quarter)
    boards = [chain.pose(to_rad(o.deg, gains=gains)) @ cam @ o.board for o in obs]
    spread = [np.median([np.linalg.norm(a[:3, 3] - b[:3, 3]) for b in boards]) for a in boards]
    best = boards[int(np.argmin(spread))]
    x = np.zeros(N_GLOBAL + 4 * len(obs))
    x[3:6], x[6:9], x[14:18] = Rotation.from_matrix(best[:3, :3]).as_rotvec(), best[:3, 3], gains
    return x


def fit(obs, chain, focal):
    """Renvoie (paramètres, écarts-types des N_GLOBAL premiers, quart de tour de l'image)."""
    s = stack(obs, focal)
    # gains de départ : ceux que montre la rotation de la caméra, quand elle les donne à mieux que 3 %
    seen, error = rotation_gains(obs)
    gains = np.where(error < 0.03, seen, 1.0)
    # le sens de montage de la caméra n'est pas connu : on essaie les quatre orientations de l'image, sans
    # corriger les lectures ni les gains, puis on ajuste tout à partir de la meilleure
    tries = []
    for quarter in range(4):
        x, sol = solve(start(obs, chain, quarter, gains), np.arange(14), s, chain, quarter)
        tries.append((sol.cost, quarter, x))
    _, quarter, x = min(tries, key=lambda t: t[:2])
    x, sol = solve(x, np.arange(len(x)), s, chain, quarter)
    # incertitude : covariance de l'ajustement, élargie si les écarts dépassent les incertitudes supposées
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * max(1.0, 2 * sol.cost / (len(sol.fun) - len(x)))
    return x, np.sqrt(np.diag(cov)[:N_GLOBAL]), quarter


def report(x, std, quarter, obs, chain, focal, floor, out):
    print("servo   gain    +/-    angle at the model's zero [deg]   +/-     (q [rad] = gain * (deg - zero) * pi / 180)")
    rows = [[1, "90.00", "", "%.4f" % x[14], "%.4f" % std[14]]]
    print("  1    %.3f   %.3f    90 (not measurable, see README.md)" % (x[14], std[14]))
    for k, sid in enumerate(FITTED):
        print("  %d    %.3f   %.3f    %6.2f                           %.2f" % (sid, x[15 + k], std[15 + k], 90 + x[k], std[k]))
        rows.append([sid, "%.2f" % (90 + x[k]), "%.2f" % std[k], "%.4f" % x[15 + k], "%.4f" % std[15 + k]])
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["servo", "deg0", "deg0_std", "gain", "gain_std"]] + rows)
    print("wrote %s" % out)
    seen, error = rotation_gains(obs)
    if fine(obs) and np.isfinite(seen).any():  # au degré près, les petits mouvements des autres servos faussent ce calcul
        print("gains from the camera's rotation alone, without the arm model: "
              + ", ".join("servo %d %.3f +/- %.3f" % (i + 1, g, e) for i, (g, e) in enumerate(zip(seen, error))
                          if np.isfinite(g)))

    # contrôles : le modèle doit expliquer les images, la mire être à plat, la caméra près de sa place dans le modèle 3D
    corners = residuals(x, stack(obs, focal), chain, quarter)[:2 * sum(len(o.obj) for o in obs)] * PIXEL_NOISE
    pixels = np.sqrt(2 * np.mean(corners ** 2))
    fixes = np.sqrt(np.mean(x[N_GLOBAL:].reshape(-1, 4) ** 2, axis=0))
    print("board corners predicted within %.2f px; servo readings corrected by %s deg rms (servos 1 to 4)"
          % (pixels, ", ".join("%.2f" % f for f in fixes)))
    normal = Rotation.from_rotvec(x[3:6]).as_matrix()[:, 2]
    tilt = np.degrees(np.arccos(min(1.0, abs(normal[2]))))
    print("board: tilted by %.1f deg from the base plane, %.0f mm %s the model's table"
          % (tilt, abs(x[8] - floor) * 1000, "above" if x[8] > floor else "below"))
    shift = x[11:14] * 1000
    print("camera: lens %+.0f mm along the housing, %+.0f mm forward, %+.0f mm away from the arm; "
          "image turned by %.0f deg, yaw %+.1f deg" % (shift[0], shift[1], shift[2], (90 * quarter + x[9]) % 360, x[10]))
    if pixels > 1.5 or fixes.max() > 0.5:
        print("warning: the model does not explain the images well: check --square, the camera intrinsics, "
              "and that nothing moved during the capture")
    if tilt > 2:
        print("warning: the board is not parallel to the base (%.1f deg): check that both lie flat on the same table" % tilt)
    if max(abs(shift[1]), abs(shift[2])) > 10:
        print("warning: the camera is not where the 3D model puts it: the servo 4 zero relies on that mounting")
    repeats(obs, chain, x, quarter)


def repeats(obs, chain, x, quarter):
    """Le bras revient-il au même endroit ? Compare les images des poses de base (a0_*, b0_* ...), prises entre
    deux balayages avec les mêmes consignes : position de la caméra d'après la mire seule."""
    groups = {}
    for o in obs:
        if re.fullmatch(r"[a-z]0_\d+", o.pose):
            groups.setdefault(o.pose[0], []).append(inverse(o.board)[:3, 3])
    spread = [np.linalg.norm(np.array(g) - np.mean(g, axis=0), axis=1).max() * 1000 for g in groups.values() if len(g) > 1]
    if spread:
        print("repeatability: back at a base pose, the camera is within %.1f mm of where it was" % max(spread))
        if max(spread) > 2:
            print("warning: the arm does not come back to the same place: the base moved, or a servo has play")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(HERE, "data"), help="folder written by capture_robot.py")
    parser.add_argument("--xml", default=os.path.join(HERE, "..", "3DModels", "dofbot_pro.xml"))
    parser.add_argument("--square", type=float, default=SQUARE * 1000,
                        help="side of one printed square, measured with a ruler [mm]")
    parser.add_argument("--board", action="store_true", help="write the board to print (charuco_A4.pdf) and exit")
    parser.add_argument("--plan", action="store_true", help="recompute the capture poses (poses.csv) and exit")
    parser.add_argument("--board-at", type=float, nargs=2, default=[0.02, 0.39], metavar=("X", "Y"),
                        help="--plan: centre of the board in the base frame [m]; +y is the front of the arm, "
                             "the base plate extends to the back (-y). Default: where the first capture found it")
    args = parser.parse_args()

    if args.board:
        return write_board(os.path.join(HERE, "charuco_A4.pdf"))
    chain = Chain(args.xml)
    floor = chain.model.geom("floor").pos[2]
    if args.plan:
        return write_poses(os.path.join(HERE, "poses.csv"), plan(chain, args.board_at))

    obs, size = load_images(args.data, charuco(args.square / 1000))
    print("%d images with the board; servo readings %s"
          % (len(obs), "in 1/12 deg steps" if fine(obs) else "in whole degrees (coarser results)"))
    if len(obs) < 8:
        sys.exit("not enough images to fit: check that the board is in view (capture_robot.py --check)")
    K, dist = load_intrinsics(args.data, obs, size)
    prepare(obs, K, dist)
    focal = (K[0, 0] + K[1, 1]) / 2
    x, std, quarter = fit(obs, chain, focal)
    report(x, std, quarter, obs, chain, focal, floor, os.path.join(args.data, "offsets.csv"))
    # contrôle croisé : une image sur deux, ajustées séparément, doivent donner les mêmes résultats. Le +/-
    # ci-dessus suppose que le modèle du bras est exact ; ce contrôle montre ce qu'il en est vraiment.
    halves = np.array([fit(obs[k::2], chain, focal)[0][:N_GLOBAL] for k in (0, 1)])
    print("cross-check, even and odd images fitted apart: zeros %s and %s deg, gains %s and %s"
          % (*(np.round(90 + h[:3], 2) for h in halves), *(np.round(h[14:18], 3) for h in halves)))
    zeros, gains = np.abs(halves[0, :3] - halves[1, :3]) / 2, np.abs(halves[0, 14:18] - halves[1, 14:18]) / 2
    if zeros.max() > 1.0 or gains.max() > 0.02:
        print("warning: the two halves disagree: the zeros are only known within about +/- %s deg and the gains "
              "within +/- %s (see README.md)" % (np.round(zeros, 1), np.round(gains, 3)))


if __name__ == "__main__":
    main()
