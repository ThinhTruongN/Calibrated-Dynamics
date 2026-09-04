from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "assets" / "cart_triple_pendulum.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

cart_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "cart_slide"
)
cart_qpos_index = int(model.jnt_qposadr[cart_joint_id])
cart_qvel_index = int(model.jnt_dofadr[cart_joint_id])
data.qpos[1:] = [0.04, -0.03, 0.02]
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "side"
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id

    while viewer.is_running() and data.time < 20.0:
        step_start = time.perf_counter()

        target_position = 1.6 * np.sin(1.2 * data.time)
        target_velocity = 1.6 * 1.2 * np.cos(1.2 * data.time)
        position_error = target_position - data.qpos[cart_qpos_index]
        velocity_error = target_velocity - data.qvel[cart_qvel_index]
        data.ctrl[0] = np.clip(
            7.0 * position_error + 2.5 * velocity_error,
            model.actuator_ctrlrange[0, 0],
            model.actuator_ctrlrange[0, 1],
        )

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining = model.opt.timestep - (time.perf_counter() - step_start)
        if remaining > 0.0:
            time.sleep(remaining)
