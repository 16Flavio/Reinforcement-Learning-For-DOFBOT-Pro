"""Photographie la mire avec la caméra du DOFBOT Pro dans une série de poses, pour fit_offsets.py.

Sur le robot, il faut les dossiers calib/ et sysid/ (ce script réutilise la classe Arm de sysid/record_robot.py),
Python 3 avec Arm_Lib, numpy et OpenCV. Le pilote de la caméra doit tourner dans un autre terminal :
    roslaunch orbbec_camera dabai_dcw2.launch         # ROS 1 (Noetic)
    ros2 launch orbbec_camera dabai_dcw2.launch.py    # ROS 2
puis :
    python3 capture_robot.py --check      # va à la première pose, enregistre data/check.png et revient
    python3 capture_robot.py              # toutes les poses de poses.csv (~6 min)
    python3 capture_robot.py --camera 0   # sans ROS : lit /dev/video0 directement avec OpenCV
    python3 capture_robot.py --dry-run    # sans robot, sur le PC : faux servos et caméra simulée

Recopier ensuite le dossier data/ à côté de fit_offsets.py. Fichiers écrits dans --out :
    img_<pose>.png    image de chaque pose
    capture.csv       pose, image, cmd1..cmd6 (consignes), lo1..lo6 et hi1..hi6 (plus petit et plus grand angle relus),
                      fine1..fine6 (angle relu sans l'arrondi d'Arm_Lib, vide si la carte ne le donne pas)
    intrinsics.json   matrice K et distorsion D de la caméra, si le pilote ROS les publie
"""
import argparse
import csv
import json
import math
import os
import signal
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sysid"))
import record_robot as rr  # noqa: E402

SLOW_MS = rr.SLOW_MS  # durée [ms] d'un grand déplacement
SETTLE = 1.0  # attente [s] après l'arrivée dans une pose, pour que le bras ne vibre plus
READS = 5     # lectures de chaque servo avant la photo, et autant après
TOPIC = "/camera/color/image_raw"
LAUNCH = ["roslaunch orbbec_camera dabai_dcw2.launch", "ros2 launch orbbec_camera dabai_dcw2.launch.py"]  # ROS 1, ROS 2
MARKERS = 17  # marqueurs de la mire de fit_offsets.py (DICT_4X4_50)
# Position brute d'un servo à 0° et en bout de course, et course [deg], dans Arm_Lib : servo 5, puis les autres (clé 0).
COUNTS = {5: (380, 3700, 270), 0: (900, 3100, 180)}


class RosCamera:
    """Image couleur publiée par le pilote ROS de la caméra, avec ROS 1 (rospy) ou ROS 2 (rclpy) selon ce que
    le terminal connaît."""

    def __init__(self, topic):
        try:
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError:
            sys.exit("ROS is not set up in this terminal: run `source /opt/ros/<distro>/setup.bash` first, "
                     "or use --camera <index>.")
        self.topic, self.image, self.info, self.count = topic, None, None, 0
        info = topic.rsplit("/", 1)[0] + "/camera_info"
        try:
            import rospy
        except ImportError:
            rospy = None
        if rospy:  # ROS 1 : rospy reçoit les messages dans ses propres fils d'exécution
            import rosgraph
            if not rosgraph.is_master_online():  # sans cela, init_node attendrait le maître indéfiniment
                sys.exit("No ROS master: start the camera driver first (%s)." % LAUNCH[0])
            rospy.init_node("calib_capture", anonymous=True, disable_signals=True)
            rospy.Subscriber(topic, Image, self._on_image, queue_size=1, buff_size=1 << 24)
            rospy.Subscriber(info, CameraInfo, self._on_info, queue_size=1)
            self.wait = time.sleep
        else:  # ROS 2 : il faut faire tourner le nœud pour recevoir
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            rclpy.init()
            node = rclpy.create_node("calib_capture")
            node.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
            node.create_subscription(CameraInfo, info, self._on_info, qos_profile_sensor_data)
            self.wait = lambda seconds: rclpy.spin_once(node, timeout_sec=seconds)
        self.launch = LAUNCH[0 if rospy else 1]

    def _on_image(self, msg):
        self.image = msg
        self.count += 1

    def _on_info(self, msg):
        self.info = msg

    def frame(self):
        """Image prise après l'appel : on laisse passer celles qui étaient déjà en route."""
        target, deadline = self.count + 3, time.time() + 5
        while self.count < target:
            if time.time() > deadline:
                sys.exit("No image on %s: is `%s` running? If it is, look for the topic of the colour image "
                         "in the list of topics and pass it with --topic." % (self.topic, self.launch))
            self.wait(0.1)
        msg = self.image
        rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        if msg.encoding in ("rgb8", "bgr8"):
            image = rows[:, :3 * msg.width].reshape(msg.height, msg.width, 3)
            return image[:, :, ::-1].copy() if msg.encoding == "rgb8" else image.copy()
        if msg.encoding == "mono8":
            return rows[:, :msg.width].copy()
        sys.exit("Image encoding %s is not handled: use --camera <index> instead." % msg.encoding)

    def intrinsics(self):
        if self.info is None:
            return None
        K, D = (self.info.K, self.info.D) if hasattr(self.info, "K") else (self.info.k, self.info.d)  # ROS 1, ROS 2
        return {"width": self.info.width, "height": self.info.height,
                "K": [float(v) for v in K], "D": [float(v) for v in D]}


class CvCamera:
    """Caméra lue directement par OpenCV (/dev/video<index>), sans ROS."""

    def __init__(self, index, size):
        self.cap = cv2.VideoCapture(index)
        if size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        if not self.cap.isOpened():
            sys.exit("Cannot open camera %d: try another index, or stop the program that is using it." % index)

    def frame(self):
        for _ in range(5):  # vide les images en attente dans le pilote
            self.cap.grab()
        ok, image = self.cap.read()
        if not ok:
            sys.exit("The camera returned no image.")
        return image

    def intrinsics(self):
        return None


def angle(sid, count, whole=False):
    """Angle [deg] qu'Arm_Lib associe à la position brute d'un servo ; whole=True l'arrondit comme Arm_Lib, qui
    tronque puis retourne les servos 2 à 4."""
    lo, hi, span = COUNTS.get(sid, COUNTS[0])
    deg = span * (count - lo) / (hi - lo)
    if whole:
        deg = int(deg)
    return 180 - deg if sid in (2, 3, 4) else deg


def fine_read(arm, sid):
    """Angle d'un servo [deg] au douzième de degré, ou None : la même demande à la carte que
    Arm_Device.Arm_serial_servo_read, sans l'arrondi au degré qui suit."""
    def ask():
        bus, addr = arm.dev.bus, arm.dev.addr
        bus.write_byte_data(addr, 0x30 + sid, 0)
        time.sleep(0.003)
        word = bus.read_word_data(addr, 0x30 + sid)
        return (word >> 8 & 0xff) | (word << 8 & 0xff00)

    try:
        count = arm._call(False, ask) or arm._call(False, ask, gap=0.0)  # même reprise que Arm.read
    except Exception:  # autre version d'Arm_Lib, ou erreur du bus
        return None
    return angle(sid, count) if count else None


def fine_works(arm):
    """Les lectures fines disent-elles la même chose qu'Arm_Lib, à son arrondi près, sur les six servos ?"""
    fine = [fine_read(arm, sid) for sid in range(1, 7)]
    whole = read_all(arm)
    return all(f is not None and w is not None and abs(f - w) <= 1.5 for f, w in zip(fine, whole))


class FakeDevice(rr.FakeDevice):
    """Faux servos de record_robot.py, avec trois traits du vrai bras : il s'affaisse sous son poids, Arm_Lib
    arrondit l'angle relu vers le bas pour les servos 1, 5 et 6 et vers le haut pour les servos 2 à 4, et la
    carte donne la position brute des servos à qui la demande comme Arm_Lib (bus, addr)."""
    SAG = [0.0, 2.0, 1.2, 0.5, 0.0, 0.0]  # affaissement [deg] de chaque servo quand son segment est à l'horizontale

    def __init__(self):
        super().__init__()
        self.bus, self.addr, self.asked = self, 0x15, None

    def true(self):
        """Angles réels des six servos [deg]."""
        lean = np.cumsum([0.0] + [p - 90 for p in self.pos[1:4]] + [0, 0])  # inclinaison du segment porté par chaque servo
        return [p + s * math.sin(math.radians(a)) + 0.08 for p, s, a in zip(self.pos, self.SAG, lean)]

    def count(self, sid):
        """Position brute du servo, telle que la carte la renvoie."""
        deg = self.true()[sid - 1]
        lo, hi, span = COUNTS.get(sid, COUNTS[0])
        return int(lo + (hi - lo) * (180 - deg if sid in (2, 3, 4) else deg) / span)

    def write_byte_data(self, addr, register, value):
        self.asked = register - 0x30 if self._accepted(0.001, register - 0x30) else None

    def read_word_data(self, addr, register):
        time.sleep(0.001)
        count = self.count(register - 0x30) if self.asked == register - 0x30 else 0
        return (count >> 8 & 0xff) | (count << 8 & 0xff00)

    def Arm_serial_servo_read(self, sid):
        return angle(sid, self.count(sid), whole=True) if self._accepted(0.004, sid) else None


class FakeCamera:
    """Caméra simulée pour --dry-run : dessine la mire comme la verrait un bras dont les servos 2 à 4 sont
    décalés de OFFSETS et dont les servos tournent GAINS fois ce qu'ils relisent, avec une caméra et une mire
    placées à peu près, mais pas exactement, comme prévu."""
    OFFSETS = [2.0, -1.5, 1.0]                  # décalages [deg] que fit_offsets.py doit retrouver
    GAINS = [0.97, 1.05, 1.0, 1.03]             # gains des servos 1 à 4, à retrouver aussi
    MOUNT = [1.0, -0.8, -0.010, 0.002, -0.001]  # écart au montage nominal (paramètres de Chain.camera)
    QUARTER = 2                                 # image à l'envers
    BOARD = (0.03, 0.38, 4.0)                   # centre de la mire [m] et rotation [deg] sur la table
    FOCAL, SIZE = 480.0, (640, 480)
    PX = 100                                    # pixels par case de la mire avant projection

    def __init__(self, device):
        import fit_offsets as fo
        from scipy.spatial.transform import Rotation
        self.fo, self.device = fo, device
        self.chain = fo.Chain(os.path.join(HERE, "..", "3DModels", "dofbot_pro.xml"))
        self.texture = fo.charuco().generateImage((fo.SQUARES[0] * self.PX, fo.SQUARES[1] * self.PX))
        # repère de la mire : origine au coin haut gauche de l'image imprimée, x à droite, y vers le bas, z dans la table
        R = Rotation.from_euler("z", self.BOARD[2], degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        centre = [self.BOARD[0], self.BOARD[1], fo.BOARD_Z]
        self.board = fo.rigid(R, centre - R @ np.append(np.array(fo.SQUARES) * fo.SQUARE / 2, 0))
        self.K = np.array([[self.FOCAL, 0, self.SIZE[0] / 2], [0, self.FOCAL, self.SIZE[1] / 2], [0, 0, 1]])
        self.rng = np.random.default_rng(0)

    def frame(self):
        fo = self.fo
        arm4 = self.chain.pose(fo.to_rad(self.device.true(), self.OFFSETS, self.GAINS))
        T = fo.inverse(arm4 @ self.chain.camera(self.MOUNT, self.QUARTER)) @ self.board
        # un point (u, v) de la texture est en (u, v, 0) * SQUARE / PX dans le repère de la mire
        H = self.K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ np.diag([fo.SQUARE / self.PX] * 2 + [1])
        image = cv2.warpPerspective(self.texture, H, self.SIZE, borderValue=120)
        image = cv2.GaussianBlur(image, (0, 0), 0.8) + self.rng.normal(0, 3, image.shape)
        return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def intrinsics(self):
        return {"width": self.SIZE[0], "height": self.SIZE[1], "K": self.K.ravel().tolist(), "D": [0.0] * 5}


def count_markers(image):
    """Nombre de marqueurs de la mire visibles dans l'image, ou None si cet OpenCV n'a pas le module aruco."""
    try:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        if hasattr(aruco, "ArucoDetector"):
            ids = aruco.ArucoDetector(dictionary).detectMarkers(image)[1]
        else:  # OpenCV antérieur à 4.7
            ids = aruco.detectMarkers(image, dictionary)[1]
    except (AttributeError, cv2.error):
        return None
    return 0 if ids is None else len(ids)


def read_all(arm):
    rr.check_stop()
    return [arm.read(sid) for sid in range(1, 7)]


def shoot(arm, camera, pose, path, fine=False):
    """Va dans la pose, photographie la mire et relit les servos avant et après la photo.
    Renvoie (plus petits angles relus, plus grands angles relus, moyenne des lectures fines, bilan affiché)."""
    # un petit pas d'un balayage n'a pas besoin des 2 s d'un grand déplacement
    rr.SLOW_MS = int(min(SLOW_MS, max(500, 40 * max(abs(a - b) for a, b in zip(pose, arm.at)))))
    rr.goto(arm, pose)
    arm.at = pose
    time.sleep(SETTLE)
    def cycle():
        reads.append(read_all(arm))
        if fine:
            fines.append([fine_read(arm, sid) for sid in range(1, 7)])

    reads, fines = [], []
    for _ in range(READS):
        cycle()
    image = camera.frame()
    for _ in range(READS):
        cycle()
    cv2.imwrite(path, image)
    lo, hi, mean = [], [], []
    for values in zip(*reads):
        values = [v for v in values if v is not None]
        lo.append(min(values) if values else "")
        hi.append(max(values) if values else "")
    for values in zip(*fines):
        values = [v for v in values if v is not None]
        mean.append("%.3f" % (sum(values) / len(values)) if values else "")
    seen = count_markers(image)
    summary = "read %s" % (mean[:4] if fine else hi[:4]) + ("" if seen is None else ", %d of %d markers seen" % (seen, MARKERS))
    return lo, hi, mean or [""] * 6, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--poses", default=os.path.join(HERE, "poses.csv"), help="poses written by fit_offsets.py --plan")
    parser.add_argument("--out", default=os.path.join(HERE, "data"), help="output folder")
    parser.add_argument("--camera", default="ros", help="'ros' (default) or the index of a video device read by OpenCV")
    parser.add_argument("--topic", default=TOPIC, help="ROS image topic (default: %s)" % TOPIC)
    parser.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), help="image size asked from a video device")
    parser.add_argument("--check", action="store_true", help="only shoot the first pose, into check.png")
    parser.add_argument("--dry-run", action="store_true", help="use fake servos and a simulated camera")
    args = parser.parse_args()

    with open(args.poses, newline="") as f:
        poses = [(row["pose"], [int(row["s%d" % i]) for i in range(1, 7)]) for row in csv.DictReader(f)]
    os.makedirs(args.out, exist_ok=True)
    if args.dry_run:
        global SLOW_MS, SETTLE
        SLOW_MS, SETTLE = 300, 0.1
        arm = rr.Arm(FakeDevice())
        camera = FakeCamera(arm.dev)
        print("dry run: simulated zeros of servos 2, 3, 4 = %s deg, gains of servos 1 to 4 = %s"
              % ([90 + v for v in FakeCamera.OFFSETS], FakeCamera.GAINS))
    else:
        camera = RosCamera(args.topic) if args.camera == "ros" else CvCamera(int(args.camera), args.size)
        print("camera: %dx%d image" % camera.frame().shape[1::-1])  # s'arrête ici, avant tout mouvement, sans image
        arm = rr.Arm()
        input("The arm is going to move over the board: clear its workspace and hold its base down.\n"
              "The factory app must be stopped first (arm-stop), it uses the same bus.\n"
              "Press Enter to start, Ctrl+C at any time to stop. ")
    signal.signal(signal.SIGINT, rr.on_sigint)
    rows = []
    try:
        rr.check_bus(arm)
        rr.goto(arm, rr.HOME)
        arm.at = rr.HOME
        fine = fine_works(arm)
        print("fine readings: %s" % ("on (1/12 deg steps)" if fine else
                                     "OFF, this Arm_Lib does not give them: whole degrees only, results will be coarser"))
        if args.check:
            path = os.path.join(args.out, "check.png")
            print("pose %s: %s" % (poses[0][0], shoot(arm, camera, poses[0][1], path, fine)[3]))
            print("wrote %s: the board should fill most of it" % path)
            return
        for name, pose in poses:
            image = "img_%s.png" % name
            lo, hi, mean, summary = shoot(arm, camera, pose, os.path.join(args.out, image), fine)
            rows.append([name, image] + pose + lo + hi + mean)
            print("pose %s: %s" % (name, summary))
    except KeyboardInterrupt:
        del rr.STOP[:]  # pour que le retour en HOME ci-dessous aille au bout
        print("\nstopped")
    finally:
        if rows:  # même après un arrêt : les poses déjà prises restent utilisables
            rr.save(os.path.join(args.out, "capture.csv"),
                    ["pose", "image"] + ["%s%d" % (k, i) for k in ("cmd", "lo", "hi", "fine") for i in range(1, 7)], rows)
            info = camera.intrinsics()
            if info:
                with open(os.path.join(args.out, "intrinsics.json"), "w") as f:
                    json.dump(info, f)
        rr.SLOW_MS = SLOW_MS
        rr.goto(arm, rr.HOME)


if __name__ == "__main__":
    main()
