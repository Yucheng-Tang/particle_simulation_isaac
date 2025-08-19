# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Granular
#
# Shows how to set up a particle-based granular material model using the
# wp.sim.ModelBuilder().
#
###########################################################################

import warp as wp
import warp.sim
import warp.sim.render
from warp.sim.model import PARTICLE_FLAG_ACTIVE
import numpy as np
import logging

logger = logging.getLogger(__name__)
log_filename = "partnums.log"
logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        # handlers=[
        # logging.FileHandler(log_filename)
        # ]
)


class Example:
    def __init__(self, stage_path="scratch_injection.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.1

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        builder.add_particle_grid(
            dim_x=16,
            dim_y=32,
            dim_z=16,
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(5.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

        self.model = builder.finalize()
        self.model.particle_kf = 25.0

        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        # self.use_cuda_graph = wp.get_device().is_cuda
        self.use_cuda_graph = False
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph



    def inject_particles(self, pos=wp.vec3(100.0, 2000.0, 0.0), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.1, radius=None):
        dev = self.model.device
        old_n = self.state_0.particle_count
        new_n = old_n + 1

        # 1) Update Model arrays (per-particle params) and particle_count
        # pos/vel in model are only used if you later call model.state(); we keep them consistent anyway
        def append_wparray(arr, np_val, dtype):
            if arr is None:
                base = np.array([], dtype=np_val.dtype).reshape(0, *np_val.shape[1:])
            else:
                base = arr.numpy()
            new_np = np.append(base, np_val, axis=0) if base.ndim == 2 else np.append(base, np_val)
            return wp.array(new_np, dtype=dtype, device=dev)

        if radius is None:
            radius = self.model.particle_radius.numpy()[-1] if self.model.particle_radius is not None else self.radius

        self.model.particle_q  = append_wparray(self.model.particle_q,  np.array([[pos[0], pos[1], pos[2]]], dtype=np.float32), wp.vec3)
        self.model.particle_qd = append_wparray(self.model.particle_qd, np.array([[vel[0], vel[1], vel[2]]], dtype=np.float32), wp.vec3)
        self.model.particle_mass = append_wparray(self.model.particle_mass, np.array([mass], dtype=np.float32), wp.float32)
        self.model.particle_inv_mass = append_wparray(self.model.particle_inv_mass, np.array([0.0 if mass == 0.0 else 1.0/mass], dtype=np.float32), wp.float32)
        self.model.particle_radius = append_wparray(self.model.particle_radius, np.array([radius], dtype=np.float32), wp.float32)
        self.model.particle_flags = append_wparray(self.model.particle_flags, np.array([int(PARTICLE_FLAG_ACTIVE.value)], dtype=np.uint32), wp.uint32)
        self.model.particle_max_radius = float(max(self.model.particle_max_radius, radius))
        self.model.particle_count = new_n  # critical for kernel dim

        # 2) Expand state_0 arrays to new length (keep previous data)
        q0  = self.state_0.particle_q.numpy()
        qd0 = self.state_0.particle_qd.numpy()
        f0  = self.state_0.particle_f.numpy()

        q0  = np.vstack([q0,  np.array([[pos[0], pos[1], pos[2]]], dtype=np.float32)])
        qd0 = np.vstack([qd0, np.array([[vel[0], vel[1], vel[2]]], dtype=np.float32)])
        f0  = np.vstack([f0,  np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])

        self.state_0.particle_q  = wp.array(q0,  dtype=wp.vec3, device=dev)
        self.state_0.particle_qd = wp.array(qd0, dtype=wp.vec3, device=dev)
        self.state_0.particle_f  = wp.array(f0,  dtype=wp.vec3, device=dev)

        # 3) Ensure state_1 arrays are same size (fresh storage for outputs)
        self.state_1.particle_q  = wp.empty_like(self.state_0.particle_q)
        self.state_1.particle_qd = wp.empty_like(self.state_0.particle_qd)
        self.state_1.particle_f  = wp.zeros_like(self.state_0.particle_qd)

        logger.info(f"state.particle_q:{self.state_0.particle_q.shape}")
        logger.info(f"Injected. model.particle_count={self.model.particle_count}, state size={self.state_0.particle_count}\n")
        

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.inject_particles()
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            self.state_0.clear_forces()
            
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        # print("Number of particles: ", self.state_0.particle_q.size)
        with wp.ScopedTimer("step"):
            # self.inject_particles()
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="scratch_injection.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=400, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
