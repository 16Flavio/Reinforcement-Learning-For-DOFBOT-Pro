"""Ajuste les paramètres des servos de dofbot_pro.xml sur les CSV enregistrés par record_robot.py.

    uv run python sysid/fit_xml.py                   # tous les servos trouvés dans sysid/data, puis met à jour le XML
    uv run python sysid/fit_xml.py --servos 1 --plot --dry-run

Pour chaque servo, les consignes enregistrées sont rejouées dans MuJoCo et kp, kv (actionneur), damping,
armature, frictionloss (articulation) ainsi qu'un retard pur sont ajustés jusqu'à ce que l'angle simulé suive
l'angle mesuré. Les valeurs sont écrites sur les éléments <joint> et <position> du XML (l'original est gardé
en dofbot_pro.xml.bak), le retard et les erreurs dans <data>/fit_results.csv. limits.csv, s'il existe, fixe
range et ctrlrange.

forcerange n'est PAS ajusté : les positions seules ne distinguent pas un servo fort d'un servo faible ayant
moins de frottement et d'inertie, donc la limite de couple fixe l'échelle des autres valeurs. Mettre le couple
de blocage mesuré dans le XML avant de lancer l'ajustement.
"""
import argparse
import csv
import glob
import os
import shutil
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))

# servo : (articulation, actionneur, deg0, pente) avec q [rad] = pente * (deg - deg0).
# Servos 1 à 5 : conversion du pilote MoveIt de Yahboom (deg = q * 180 / pi + 90).
# Pince : 30° (ouverte) -> q = 0 et 180° (fermée) -> q = -pi/2, soit 0,6° d'articulation par degré de servo
# comme chez Yahboom, mais dans le sens ouvert/fermé du modèle MuJoCo -- à vérifier sur le vrai bras.
DEG = np.pi / 180
SERVOS = {1: ("Arm1_Joint", "Arm1", 90, DEG), 2: ("Arm2_Joint", "Arm2", 90, DEG), 3: ("Arm3_Joint", "Arm3", 90, DEG),
          4: ("Arm4_Joint", "Arm4", 90, DEG), 5: ("Arm5_Joint", "Arm5", 90, DEG),
          6: ("grip_joint", "grip", 30, -0.6 * DEG)}

NAMES = ["kp", "kv", "damping", "armature", "frictionloss", "delay"]
LOW = np.array([1.0, 1e-3, 1e-3, 1e-5, 1e-4, 0.0])
HIGH = np.array([1e3, 1e2, 1e2, 1.0, 5.0, 0.2])
# second point de départ, essayé en plus des valeurs actuelles du XML (même retard initial)
STIFF = [80.0, 1.0, 0.3, 0.02, 0.02]


def to_rad(sid, deg):
    return SERVOS[sid][3] * (np.asarray(deg, dtype=float) - SERVOS[sid][2])


def to_deg(sid, rad):
    return SERVOS[sid][2] + rad / SERVOS[sid][3]


def encode(p):  # variables de l'optimiseur : log des cinq gains, retard en secondes
    return np.append(np.log(p[:5]), p[5])


def decode(x):
    return np.append(np.exp(x[:5]), x[5])


def get_params(model, jid, aid):
    dof = model.jnt_dofadr[jid]
    p = [model.actuator_gainprm[aid, 0], -model.actuator_biasprm[aid, 2],
         model.dof_damping[dof], model.dof_armature[dof], model.dof_frictionloss[dof], 0.0]
    return np.clip(p, LOW, HIGH)


def set_params(model, data, jid, aid, p):
    kp, kv, damping, armature, frictionloss = p[:5]
    dof = model.jnt_dofadr[jid]
    model.actuator_gainprm[aid, 0] = kp
    model.actuator_biasprm[aid, 1:3] = -kp, -kv
    model.dof_damping[dof] = damping
    model.dof_armature[dof] = armature
    model.dof_frictionloss[dof] = frictionloss
    mujoco.mj_setConst(model, data)


def load_runs(model, data, sid, files):
    """Lit les enregistrements d'un servo et prépare la consigne de chaque essai sur la grille de temps de la simulation."""
    by_run = {}
    for path in files:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                by_run.setdefault(row["run"], []).append(row)

    aid = model.actuator(SERVOS[sid][1]).id
    dt = model.opt.timestep
    runs = []
    for name, rows in by_run.items():
        t = np.array([float(r["t"]) for r in rows])
        cmd = np.array([float(r["cmd"]) for r in rows])
        t_cmd = np.array([float(r["t_cmd"]) for r in rows])  # instant d'envoi de la consigne de chaque ligne
        pos = np.array([float(r["pos"]) if r["pos"] else np.nan for r in rows])
        ctrl0 = np.array([to_rad(s, rows[0]["hold%d" % s]) for s in range(1, 7)])
        ctrl0[aid] = to_rad(sid, cmd[0])
        t_sim = t[0] + dt * np.arange(int(np.ceil((t[-1] - t[0]) / dt)) + 1)
        held = np.clip(np.searchsorted(t_cmd, t_sim[:-1], side="right") - 1, 0, len(t) - 1)

        # état initial : la pose maintenue, qu'on laisse se stabiliser (tringlerie de la pince, affaissement sous la gravité)
        mujoco.mj_resetData(model, data)
        for s in range(1, 7):
            data.qpos[model.joint(SERVOS[s][0]).qposadr] = ctrl0[model.actuator(SERVOS[s][1]).id]
        data.ctrl[:] = ctrl0
        for _ in range(int(0.5 / dt)):
            mujoco.mj_step(model, data)

        runs.append(SimpleNamespace(name=name, t=t, cmd=cmd, pos=pos, ok=~np.isnan(pos), t_sim=t_sim,
                                    ctrl=to_rad(sid, cmd[held]), ctrl0=ctrl0, qpos0=data.qpos.copy()))
    return runs


def simulate(model, data, sid, runs, p):
    """Rejoue les consignes avec les paramètres p ; renvoie, par essai, l'angle simulé [deg] aux instants de mesure."""
    jid, aid = model.joint(SERVOS[sid][0]).id, model.actuator(SERVOS[sid][1]).id
    adr = model.jnt_qposadr[jid]
    set_params(model, data, jid, aid, p)
    out = []
    for r in runs:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = r.qpos0
        data.ctrl[:] = r.ctrl0
        q = np.empty(len(r.ctrl) + 1)
        q[0] = data.qpos[adr]
        for k, u in enumerate(r.ctrl):
            data.ctrl[aid] = u
            mujoco.mj_step(model, data)
            q[k + 1] = data.qpos[adr]
        out.append(to_deg(sid, np.interp(r.t - p[5], r.t_sim, q)))
    return out


def fit(model, data, sid, runs):
    """Renvoie (paramètres, rmse [deg]) du meilleur ajustement, puis ceux des valeurs actuelles du XML."""
    measured = np.concatenate([r.pos[r.ok] for r in runs])

    def residuals(x):
        sim = np.concatenate([s[r.ok] for s, r in zip(simulate(model, data, sid, runs, decode(x)), runs)])
        return np.nan_to_num(sim - measured, nan=1e3, posinf=1e3, neginf=-1e3)

    def jacobian(x):  # différences finies avant, avec un pas assez grand pour enjamber les cassures de la simulation
        r0 = residuals(x)
        steps = np.array([0.02] * 5 + [1e-3])
        return np.stack([(residuals(x + h * np.eye(6)[i]) - r0) / h for i, h in enumerate(steps)], axis=1)

    def rmse(x):
        return float(np.sqrt(np.mean(residuals(x) ** 2)))

    jid, aid = model.joint(SERVOS[sid][0]).id, model.actuator(SERVOS[sid][1]).id
    current = get_params(model, jid, aid)
    before = rmse(encode(current))
    best = None
    for start in (np.append(current[:5], 0.02), np.append(STIFF, 0.02)):
        res = least_squares(residuals, encode(start), jac=jacobian, bounds=(encode(LOW), encode(HIGH)),
                            x_scale=[1, 1, 1, 1, 1, 0.02])
        if best is None or res.cost < best.cost:
            best = res
    p = decode(best.x)
    set_params(model, data, jid, aid, p)  # les servos suivants sont ajustés avec celui-ci déjà mis à jour
    return p, rmse(best.x), current, before


def plot(model, data, sid, runs, before, p, path):
    """Trace, par essai, la mesure et MuJoCo avec les paramètres d'avant (XML actuel) et d'après l'ajustement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    nrows = -(-len(runs) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 2.6 * nrows), squeeze=False)
    sims = simulate(model, data, sid, runs, before), simulate(model, data, sid, runs, p)  # le modèle garde p
    for ax, r, sim_before, sim in zip(axes.flat, runs, *sims):
        ax.plot(r.t_sim[:-1], to_deg(sid, r.ctrl), color="0.7", label="command")
        ax.plot(r.t, r.pos, ".", ms=2, color="k", label="robot")
        ax.plot(r.t, sim_before, "--", color="tab:blue", lw=1, label="MuJoCo before")
        ax.plot(r.t, sim, color="tab:red", lw=1, label="MuJoCo after")
        ax.set_title(r.name, fontsize=9)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("t [s]")
    fig.supylabel("servo %d angle [deg]" % sid)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def update_xml(path, results, limits):
    """Écrit les valeurs ajustées et les butées mesurées dans le XML, en gardant le premier original en .bak."""
    def fmt(*values):
        return " ".join("%.4g" % v for v in values)

    tree = ET.parse(path)
    root = tree.getroot()
    for sid, p in results.items():
        joint = root.find(".//joint[@name='%s']" % SERVOS[sid][0])
        actuator = root.find("./actuator/position[@name='%s']" % SERVOS[sid][1])
        actuator.set("kp", fmt(p[0]))
        actuator.set("kv", fmt(p[1]))
        joint.set("damping", fmt(p[2]))
        joint.set("armature", fmt(p[3]))
        joint.set("frictionloss", fmt(p[4]))
    for sid, (lo, hi) in limits.items():
        span = "%.4f %.4f" % (lo, hi)
        root.find(".//joint[@name='%s']" % SERVOS[sid][0]).set("range", span)
        root.find("./actuator/position[@name='%s']" % SERVOS[sid][1]).set("ctrlrange", span)
    if not os.path.exists(path + ".bak"):
        shutil.copy(path, path + ".bak")
    tree.write(path)


def load_limits(path):
    limits = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row["pos_min"] and row["pos_max"]:
                    sid = int(row["servo"])
                    limits[sid] = sorted(float(to_rad(sid, row[k])) for k in ("pos_min", "pos_max"))
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join(HERE, "data"), help="folder written by record_robot.py")
    parser.add_argument("--xml", default=os.path.join(HERE, "..", "3DModels", "dofbot_pro.xml"))
    parser.add_argument("--servos", type=int, nargs="+", default=list(SERVOS), choices=list(SERVOS))
    parser.add_argument("--plot", action="store_true",
                        help="save <data>/fit_servo<N>.png (robot vs MuJoCo before and after the fit)")
    parser.add_argument("--dry-run", action="store_true", help="fit and report, but leave the XML untouched")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    # les tests sont des mouvements libres à l'intérieur de la course : contacts et butées ne font que ralentir l'ajustement
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_LIMIT

    results, report = {}, []
    for sid in args.servos:
        files = sorted(glob.glob(os.path.join(args.data, "*_servo%d.csv" % sid)))
        if not files:
            print("servo %d: no recording in %s, skipped" % (sid, args.data))
            continue
        runs = load_runs(model, data, sid, files)
        print("servo %d: fitting on %d runs..." % (sid, len(runs)), flush=True)
        p, after, current, before = fit(model, data, sid, runs)
        results[sid] = p
        report.append([sid] + ["%.4g" % v for v in p] + ["%.2f" % before, "%.2f" % after])
        print("  " + "  ".join("%s=%.4g" % nv for nv in zip(NAMES, p)))
        print("  rmse %.2f deg -> %.2f deg" % (before, after))
        at_bound = [n for n, v, lo, hi in zip(NAMES[:5], p, LOW, HIGH) if v < lo * 1.01 or v > hi * 0.99]
        if at_bound:
            print("  warning: %s stuck at a bound, not identified by this data" % ", ".join(at_bound))
        if args.plot:
            plot(model, data, sid, runs, current, p, os.path.join(args.data, "fit_servo%d.png" % sid))

    limits = load_limits(os.path.join(args.data, "limits.csv"))
    if not results and not limits:
        return
    if results:
        with open(os.path.join(args.data, "fit_results.csv"), "w", newline="") as f:
            csv.writer(f).writerows([["servo"] + NAMES[:5] + ["delay_s", "rmse_before_deg", "rmse_after_deg"]] + report)
    if args.dry_run:
        print("dry run: %s left untouched" % args.xml)
    else:
        update_xml(args.xml, results, limits)
        print("updated %s (servos %s, limits %s)" % (args.xml, sorted(results), sorted(limits) or "none"))


if __name__ == "__main__":
    main()
