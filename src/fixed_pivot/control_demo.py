from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np


model_path = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "triple_pendulum.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Initial joint angles, measured in radians.
data.qpos[:] = [0.2, -0.1, 0.15]
mujoco.mj_forward(model, data)

step_count = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time < 20:
        step_start = time.time()

        # Temporary hand-written actions to test the motors.
        data.ctrl[:] = [
            1.5 * np.sin(data.time),
            1.0 * np.sin(1.4 * data.time + 0.5),
            0.8 * np.sin(1.8 * data.time + 1.0),
        ]

        mujoco.mj_step(model, data)
        viewer.sync()

        # Print the state once every 100 simulation steps.
        if step_count % 100 == 0:
            angles = data.qpos.copy()
            angular_velocities = data.qvel.copy()
            torques = data.ctrl.copy()

            print(f"\nTime: {data.time:.2f}")
            print("Angles:             ", np.round(angles, 3))
            print("Angular velocities: ", np.round(angular_velocities, 3))
            print("Torques:            ", np.round(torques, 3))

        step_count += 1

        remaining_time = model.opt.timestep - (time.time() - step_start)

        if remaining_time > 0:
            time.sleep(remaining_time)
