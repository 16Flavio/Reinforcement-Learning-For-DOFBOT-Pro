import mujoco, mujoco.viewer
import numpy as np
import os
from pathlib import Path

def deg_to_rad(deg):
    return deg*(np.pi/180)
def rad_to_deg(rad):
    return rad*(180/np.pi)

model = mujoco.MjModel.from_xml_path(os.path.join(Path(os.path.dirname(__file__)).parent.parent / "3DModels", "dofbot_pro.xml"))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as v:

    step = 0
    while v.is_running():
        if step <= 500:
            # On dresse le bras avec les 4 premiers servos
            data.ctrl[:4] = [deg_to_rad(0), deg_to_rad(0), deg_to_rad(0), deg_to_rad(0)]

            # On avance la simulation
            mujoco.mj_step(model, data)
            v.sync()

            print(f"Les articulations valent : {" ".join([str(rad_to_deg(rad)) for rad in data.qpos])}")
            print(f"Position de la pince : {data.site("grip_site").xpos}")
            step += 1 
        if step > 500 and step <= 1000:
            # Nouvelle pose         
            data.ctrl[2] = deg_to_rad(-45)

            # On avance la simulation
            mujoco.mj_step(model, data)
            v.sync()

            print(f"Les articulations valent : {" ".join([str(rad_to_deg(rad)) for rad in data.qpos])}")
            print(f"Position de la pince : {data.site("grip_site").xpos}")
            step += 1
