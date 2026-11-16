"""Lance des tests d'identification sur le vrai DOFBOT Pro et enregistre les mesures en CSV.

Copier ce fichier sur le robot (il ne lui faut que Python 3 et la librairie Arm_Lib de Yahboom) puis lancer :
    python3 record_robot.py --gentle         # à faire en premier : petits mouvements lents, pour vérifier que tout marche
    python3 record_robot.py                  # timing + échelons + chirp sur les 6 servos (~5 min)
    python3 record_robot.py --servos 2 3     # seulement certains servos
    python3 record_robot.py --tests limits   # sur demande : amène les servos 1 à 5 aux deux bouts de leur course
    python3 record_robot.py --dry-run        # sans robot : faux servos, pour essayer le script

Recopier ensuite le dossier data/ à côté de fit_xml.py. Les mesures de --gentle vont dans data_gentle/ : elles
servent à contrôler le fonctionnement, pas à ajuster le modèle (les servos n'y sont pas à vitesse maximale).
La méthode complète est décrite dans README.md.

Fichiers écrits dans --out (angles en degrés Yahboom, tels qu'envoyés à / lus depuis Arm_Lib) :
    timing.csv            durée des opérations sur le bus, lectures ratées
    steps_servo<N>.csv    run, t [s], cmd, t_cmd [s], pos, hold1..hold6 (t = instant de la lecture, t_cmd = instant d'envoi de cmd)
    chirp_servo<N>.csv    mêmes colonnes (hold* = consigne maintenue sur les six servos pendant l'essai)
    limits.csv            angle réellement atteint à chaque bout de la course
"""
import argparse
import csv
import math
import os
import signal
import statistics
import sys
import time

HOME = [90, 90, 90, 90, 90, 120]  # bras à la verticale, pince à moitié ouverte
MOVE_MS = 0      # durée donnée au servo pour une consigne de test : 0 = vitesse maximale (doc Yahboom)
SLOW_MS = 2000   # durée des déplacements entre deux tests
# servo : (centre, amplitudes des échelons) en degrés ; la plus grande amplitude est aussi parcourue d'un bout à l'autre
STEPS = {1: (90, [10, 30]), 2: (90, [10, 30]), 3: (90, [10, 30]),
         4: (90, [10, 30]), 5: (90, [10, 30]), 6: (120, [10, 30])}
CHIRP = (15, 0.2, 1.5, 10.0)  # amplitude [deg], fréquences de début et de fin [Hz], durée [s]
LIMITS = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270)}  # consignes envoyées par le test des butées
# réglages remplacés par --gentle : échelons de 10° exécutés en 0,5 s, chirp plus petit et plus lent, déplacements en 4 s
GENTLE = dict(MOVE_MS=500, SLOW_MS=4000, CHIRP=(5, 0.2, 0.5, 5.0),
              STEPS={sid: (centre, [10]) for sid, (centre, _) in STEPS.items()})
TOLERANCE = 3  # écart [deg] accepté entre la consigne et l'angle relu à la fin d'un échelon
# La carte du bras ignore, sans erreur, une opération qui suit la précédente de trop près : la lecture renvoie
# None, la consigne est perdue. Pauses [s] laissées avant chaque opération pour l'éviter :
READ_GAP = 0.005   # entre deux lectures
WRITE_GAP = 0.010  # avant et après une écriture (valeur des exemples Yahboom)
# Elle refuse aussi de relire deux fois de suite le même servo, quelle que soit la pause : voir Arm.read.


class Arm:
    """Les quelques appels à Arm_Lib utilisés ici ; adapter cette classe si votre librairie diffère.
    Chaque appel laisse d'abord reposer le bus (READ_GAP, WRITE_GAP) et note son instant dans `stamp`."""

    def __init__(self, dev=None):
        if dev is None:
            from Arm_Lib import Arm_Device
            dev = Arm_Device()
            time.sleep(0.1)
        self.dev = dev
        self.last = time.perf_counter()  # fin de la dernière opération
        self.wrote = True                # cette opération était une écriture
        self.stamp = self.last           # milieu de la dernière opération

    def _call(self, write, fn, *args, gap=None):
        if gap is None:
            gap = WRITE_GAP if write or self.wrote else READ_GAP
        time.sleep(max(0.0, self.last + gap - time.perf_counter()))
        start = time.perf_counter()
        result = fn(*args)
        self.last, self.wrote = time.perf_counter(), write
        self.stamp = (start + self.last) / 2
        return result

    def write(self, sid, deg, ms):
        self._call(True, self.dev.Arm_serial_servo_write, sid, deg, int(ms))

    def write_all(self, degs, ms):
        self._call(True, self.dev.Arm_serial_servo_write6, *degs, int(ms))

    def read(self, sid):
        """Angle en degrés, ou None si la lecture a échoué. Relire le servo qu'on vient de lire échoue toujours,
        et la demande suivante réussit : après un échec, on redemande donc aussitôt, une fois."""
        pos = self._call(False, self.dev.Arm_serial_servo_read, sid)
        if pos is None:
            pos = self._call(False, self.dev.Arm_serial_servo_read, sid, gap=0.0)
        return pos


class FakeDevice:
    """Remplaçant d'Arm_Device pour --dry-run : des servos qui tournent à vitesse constante vers leur consigne,
    sur une carte qui, comme la vraie, ignore une opération arrivant moins de BUSY s après la précédente et
    la relecture du servo qu'elle vient de lire."""
    SPEED = 300.0  # deg/s
    BUSY = 0.003

    def __init__(self):
        self.pos = [float(d) for d in HOME]
        self.target = list(self.pos)
        self.last = self.free = time.perf_counter()
        self.served = None  # servo dont la carte vient de donner l'angle

    def _accepted(self, duration, sid=None):
        """Fait durer l'opération et avancer les servos ; renvoie False si la carte l'ignore."""
        accepted = time.perf_counter() >= self.free and (sid is None or sid != self.served)
        self.served = sid if accepted else None
        time.sleep(duration)
        now = time.perf_counter()
        step = self.SPEED * (now - self.last)
        self.last = now
        self.pos = [p + max(-step, min(step, t - p)) for p, t in zip(self.pos, self.target)]
        if accepted:
            self.free = now + self.BUSY
        return accepted

    def Arm_serial_servo_write(self, sid, deg, ms):
        if self._accepted(0.0006):
            self.target[sid - 1] = float(deg)

    def Arm_serial_servo_write6(self, s1, s2, s3, s4, s5, s6, ms):
        if self._accepted(0.002):
            self.target = [float(d) for d in (s1, s2, s3, s4, s5, s6)]

    def Arm_serial_servo_read(self, sid):
        return int(self.pos[sid - 1]) if self._accepted(0.004, sid) else None


STOP = []  # non vide dès qu'un Ctrl+C a été reçu


def on_sigint(signum, frame):
    # Arm_Lib entoure ses accès au bus d'un `except:` nu qui avale KeyboardInterrupt : on mémorise
    # donc la demande d'arrêt, et check_stop() la relance en dehors de la librairie.
    STOP.append(1)
    raise KeyboardInterrupt


def check_stop():
    if STOP:
        raise KeyboardInterrupt


def check_bus(arm):
    """Relit les six servos avant tout mouvement : si le bus ne répond pas, inutile d'aller plus loin."""
    angles = []
    for sid in range(1, 7):
        reads = [p for p in (arm.read(sid) for _ in range(3)) if p is not None]
        angles.append(reads[-1] if reads else None)
    print("current angles: %s" % angles)
    if None in angles:
        sys.exit("Some servos do not answer: check the power supply, and that the factory app is stopped (arm-stop).")


def goto(arm, pose):
    check_stop()
    arm.write_all(pose, SLOW_MS)
    time.sleep(SLOW_MS / 1000 + 0.5)
    check_stop()


def save(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print("  wrote %s (%d rows)" % (path, len(rows)))


def record(arm, sid, command, duration, lead=0.3):
    """Envoie command(t) [deg] à un servo et relit sa position aussi vite que le bus le permet.
    t = 0 arrive `lead` secondes après le début ; renvoie des lignes (t, cmd, t_cmd, pos), où t est l'instant
    de la lecture et t_cmd celui où cmd a été envoyée."""
    rows, sent, t_sent = [], None, None
    t0 = time.perf_counter() + lead
    while True:
        check_stop()
        t = time.perf_counter() - t0
        if t > duration:
            return rows
        cmd = command(t)
        if cmd != sent:
            arm.write(sid, cmd, MOVE_MS)
            sent, t_sent = cmd, arm.stamp - t0
        pos = arm.read(sid)
        rows.append((arm.stamp - t0, cmd, t_sent, pos))


def step_runs(centre, amps):
    """(départ, arrivée) de chaque échelon : aller-retour de chaque côté, puis un balayage complet."""
    runs = []
    for a in amps:
        runs += [(centre, centre + a), (centre + a, centre), (centre, centre - a), (centre - a, centre)]
    a = max(amps)
    return runs + [(centre - a, centre + a), (centre + a, centre - a)]


def verdict(label, checks, arrival=True):
    """Bilan d'un test : le servo finit-il sur sa consigne, des lectures ont-elles échoué ?
    Renvoie True si tout va bien."""
    error = max(c[0] for c in checks) if arrival else 0
    missed, total = sum(c[1] for c in checks), sum(c[2] for c in checks)
    ok = error <= TOLERANCE and missed <= 0.05 * total
    arrived = "ends within %g deg of the command, " % error if arrival else ""
    print("  %s: %s%d failed reads out of %d -> %s" % (label, arrived, missed, total, "OK" if ok else "PROBLEM"))
    return ok


def test_servo(arm, sid, tests, out):
    """Échelons et chirp d'un servo ; renvoie True si les bilans sont bons."""
    centre, amps = STEPS[sid]
    header = ["run", "t", "cmd", "t_cmd", "pos"] + ["hold%d" % i for i in range(1, 7)]
    at = HOME[sid - 1]
    ok = True

    def run(rows, name, start, command, duration):
        """Enregistre un essai ; renvoie (écart final à la consigne, lectures ratées, lectures)."""
        nonlocal at
        pose = list(HOME)
        pose[sid - 1] = start
        if start != at:
            goto(arm, pose)
        recorded = record(arm, sid, command, duration)
        for t, cmd, t_cmd, pos in recorded:
            rows.append([name, "%.4f" % t, cmd, "%.4f" % t_cmd, "" if pos is None else pos] + pose)
        at = cmd
        reads = [pos for _, _, _, pos in recorded if pos is not None]
        return abs(reads[-1] - cmd) if reads else float("inf"), len(recorded) - len(reads), len(recorded)

    if "steps" in tests:
        rows = []
        checks = [run(rows, "step_%d_%d" % (a0, a1), a0, lambda t, a0=a0, a1=a1: a0 if t < 0 else a1, 1.5)
                  for a0, a1 in step_runs(centre, amps)]
        save(os.path.join(out, "steps_servo%d.csv" % sid), header, rows)
        ok = verdict("steps", checks) and ok
    if "chirp" in tests:
        amp, f0, f1, dur = CHIRP

        def chirp(t):
            if t < 0:
                return centre
            return round(centre + amp * math.sin(2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur))), 2)

        rows = []
        check = run(rows, "chirp", centre, chirp, dur)
        save(os.path.join(out, "chirp_servo%d.csv" % sid), header, rows)
        ok = verdict("chirp", [check], arrival=False) and ok  # la consigne bouge encore à la fin du chirp
    goto(arm, HOME)
    return ok


def test_timing(arm, out):
    """Chronomètre les opérations sur le bus, pauses comprises, sans bouger le bras (il est déjà en HOME)."""
    def cycle():
        arm.write_all(HOME, MOVE_MS)
        return [arm.read(sid) for sid in range(1, 7)]

    rows = []
    for name, fn, n in [("read_one", lambda: [arm.read(1)], 100),
                        ("write_one", lambda: arm.write(1, HOME[0], MOVE_MS), 100),
                        ("write_all", lambda: arm.write_all(HOME, MOVE_MS), 100),
                        ("cycle_write_all_read_all", cycle, 50)]:
        ms, reads = [], []
        for _ in range(n):
            check_stop()
            t = time.perf_counter()
            reads += fn() or []
            ms.append((time.perf_counter() - t) * 1000)
        rows.append([name, n, "%.3f" % statistics.mean(ms), "%.3f" % statistics.median(ms), "%.3f" % max(ms),
                     reads.count(None), len(reads)])
    save(os.path.join(out, "timing.csv"), ["operation", "n", "mean_ms", "median_ms", "max_ms", "failed_reads", "reads"], rows)
    for row in rows:
        if row[5] > 0.05 * row[6]:
            print("  %s: %d failed reads out of %d -> PROBLEM" % (row[0], row[5], row[6]))
    print("  full write+read cycle: %s ms median -> at most %.0f Hz of control" % (rows[-1][3], 1000 / float(rows[-1][3])))


def test_limits(arm, servos, out):
    rows = []
    for sid in servos:
        if sid not in LIMITS:
            continue
        reached = []
        for cmd in LIMITS[sid]:
            pose = list(HOME)
            pose[sid - 1] = cmd
            goto(arm, pose)
            time.sleep(1.0)
            reads = [p for p in (arm.read(sid) for _ in range(10)) if p is not None]
            reached.append("%.1f" % statistics.mean(reads) if reads else "")
        goto(arm, HOME)
        rows.append([sid, LIMITS[sid][0], reached[0], LIMITS[sid][1], reached[1]])
    save(os.path.join(out, "limits.csv"), ["servo", "cmd_min", "pos_min", "cmd_max", "pos_max"], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--servos", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], choices=range(1, 7))
    parser.add_argument("--tests", nargs="+", default=["timing", "steps", "chirp"],
                        choices=["timing", "steps", "chirp", "limits"])
    parser.add_argument("--out", help="output folder (default: data/, or data_gentle/ with --gentle, next to this file)")
    parser.add_argument("--gentle", action="store_true",
                        help="small slow moves to check that everything works; not usable for fitting")
    parser.add_argument("--dry-run", action="store_true", help="use fake servos instead of the robot")
    args = parser.parse_args()
    if args.gentle and "limits" in args.tests:
        parser.error("--gentle does not run the limits test")
    if args.gentle:
        globals().update(GENTLE)
        print("gentle mode: steps of 10 deg done in %d ms, small slow chirp; for checking only, not for fitting" % MOVE_MS)
    if not args.out:
        folder = "data_gentle" if args.gentle else "data"
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)

    os.makedirs(args.out, exist_ok=True)
    if args.dry_run:
        arm = Arm(FakeDevice())
    else:
        arm = Arm()
        input("The arm is going to move%s: clear its workspace and hold its base down.\n"
              "The factory app must be stopped first (arm-stop), it uses the same bus.\n"
              "Press Enter to start, Ctrl+C at any time to stop. " % (" slowly" if args.gentle else " fast"))
    signal.signal(signal.SIGINT, on_sigint)
    try:
        check_bus(arm)
        goto(arm, HOME)
        if "timing" in args.tests:
            print("timing")
            test_timing(arm, args.out)
        if "limits" in args.tests:
            print("limits")
            test_limits(arm, args.servos, args.out)
        if "steps" in args.tests or "chirp" in args.tests:
            failed = []
            for sid in args.servos:
                print("servo %d" % sid)
                if not test_servo(arm, sid, args.tests, args.out):
                    failed.append(sid)
            print("all servos OK" if not failed else "PROBLEM on servos %s: see the lines above" % failed)
    except KeyboardInterrupt:
        del STOP[:]  # pour que le retour en HOME ci-dessous aille au bout
        print("\nstopped")
    finally:
        goto(arm, HOME)


if __name__ == "__main__":
    main()
