import os
import numpy as np
import warp as wp
import warp.render
import warp.examples
from pxr import Usd, UsdGeom


@wp.kernel
def emit_new_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    emission_pos: wp.vec3,
    emission_vel: wp.vec3,
    seed: wp.int32,
    emission_spread: float,
    emission_vel_spread: float,
    current_particles: int,
    max_particles: int
):
    tid = wp.tid()
    if tid < current_particles or tid >= max_particles:
        return

    rng = wp.rand_init(seed, tid)
    pos_noise = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * emission_spread
    vel_noise = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * emission_vel_spread
    pos = emission_pos + pos_noise
    vel = emission_vel + vel_noise
    positions[tid] = pos
    velocities[tid] = vel

@wp.kernel
def simulate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    sim_timestep: float,
    mesh: wp.uint64
):
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, 0.0 - 9.8, 0.0) * sim_timestep - v * 0.1 * sim_timestep
    xpred = x + v * sim_timestep

    # pbd update
    v = (xpred - x) * (1.0 / sim_timestep)
    x = xpred

    max_dist = 1.5

    query = wp.mesh_query_point_sign_normal(mesh, xpred, max_dist)
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        x = p
        v = wp.vec3(0.0, 0.0, 0.0)

    positions[tid] = x
    velocities[tid] = v

class Emission:
    def __init__(self, stage_path='emission.usd'):
        self.rng = np.random.default_rng(7777)
        self.seed = 7777
        self.sim_timestep = 1.0 / 60.0
        self.sim_time = 0.0

        self.max_particles = 100_000
        self.particle_radius = 0.1
        self.positions = wp.empty(self.max_particles, dtype=wp.vec3)
        self.velocities = wp.empty(self.max_particles, dtype=wp.vec3)

        self.current_particles = 0
        self.emission_counter = 0.0
        self.emission_rate = 1_000     # particles per second
        self.emission_pos = wp.vec3(0.0, 20.0, 0.0)
        self.emission_spread = 0.5
        self.emission_vel = wp.vec3(0.0, 0.0, 0.0)
        self.emission_vel_spread = 1.0

        # create collision mesh
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        usd_scale = 10.0
        self.mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get() * usd_scale, dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    def emit_new_particles(self):
        """Emit new particles based on emission rate"""
        if self.current_particles >= self.max_particles:
            return
        self.emission_counter += self.emission_rate * self.sim_timestep

        particles_to_emit = int(self.emission_counter)
        self.emission_counter -= particles_to_emit      # account for the remainder

        wp.launch(
            kernel = emit_new_particles,
            dim = self.current_particles + particles_to_emit,
            inputs = [
                self.positions,
                self.velocities,
                self.emission_pos,
                self.emission_vel,
                self.seed,
                self.emission_spread,
                self.emission_vel_spread,
                self.current_particles,
                self.max_particles
            ]
        )
        self.current_particles += particles_to_emit

    def step(self):
        with wp.ScopedTimer("step"):
            self.emit_new_particles()
            if self.current_particles > 0:
                wp.launch(
                    kernel = simulate,
                    dim = self.current_particles,
                    inputs = [self.positions, self.velocities, self.sim_timestep, self.mesh.id]
                )
            self.sim_time += self.sim_timestep

    def render(self):
        if self.render is None:
            return
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_mesh(
                name="mesh",
                points=self.mesh.points.numpy(),
                indices=self.mesh.indices.numpy(),
                colors=(0.35, 0.55, 0.9),
            )
            self.renderer.render_points(
                name="points", points=self.positions.numpy(), radius=self.particle_radius, colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=200)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        e = Emission()
        for _ in range(args.num_frames):
            e.step()
            e.render()

        if e.renderer:
            e.renderer.save()

