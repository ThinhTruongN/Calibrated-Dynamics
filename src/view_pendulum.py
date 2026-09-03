from pathlib import Path
import time

import mujoco
import mujoco.viewer


model_path = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "single_pendulum.xml"
)

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Start the pendulum at an angle instead of perfectly downward.
data.qpos[0] = 0.7
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    while viewer.is_running() and time.time() - start_time < 20:
        step_start = time.time()

        mujoco.mj_step(model, data)
        viewer.sync()

        remaining_time = model.opt.timestep - (time.time() - step_start)

        if remaining_time > 0:
            time.sleep(remaining_time)