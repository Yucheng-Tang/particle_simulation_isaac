import os
import numpy as np
import warp as wp
import warp.render
import warp.examples
from pxr import Usd, UsdGeom
import logging

# np.set_printoptions(threshold=100, linewidth=100)

logger = logging.getLogger(__name__)
log_filename = "emission.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    # handlers=[
    # logging.FileHandler(log_filename)
    # ]
)


@wp.kernel
def emit_new_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    on_mesh: wp.array(dtype=wp.int8),
    emission_pos: wp.vec3,
    emission_vel: wp.vec3,
    seed: wp.int32,
    emission_spread: float,
    emission_vel_spread: float,
):
    tid = wp.tid()
    # part_on_mesh = -1(uninitialized), 1(True), 0(False)
    part_on_mesh = on_mesh[tid]

    # overwrite this thread to spawn new particle
    # TODO: adapt to emission rate
    if wp.abs(part_on_mesh) == 1:
        rng = wp.rand_init(seed, tid)
        pos_noise = (
            wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * emission_spread
        )
        vel_noise = (
            wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * emission_vel_spread
        )
        pos = emission_pos + pos_noise
        vel = emission_vel + vel_noise

        positions[tid] = pos
        velocities[tid] = vel
        on_mesh[tid] = wp.int8(0)

    # unfinished simulation
    else:
        assert part_on_mesh == 0
        return


@wp.kernel
def simulate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    sim_timestep: float,
    mesh: wp.uint64,
    on_mesh: wp.array(dtype=wp.int8),
    positions_on_mesh: wp.array(dtype=wp.vec3),
    num_particles_on_mesh: wp.array(dtype=int),
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

    # once a particle landed on the mesh
    # stick onto the mesh at the contacting point
    query = wp.mesh_query_point_sign_normal(mesh, xpred, max_dist)
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        x = p
        v = wp.vec3(0.0, 0.0, 0.0)
        on_mesh[tid] = wp.int8(1)
        positions_on_mesh[num_particles_on_mesh[0]] = x
        num_particles_on_mesh[0] += 1

    positions[tid] = x
    velocities[tid] = v


class Emission:
    def __init__(self, stage_path="emission.usd"):
        self.rng = np.random.default_rng(7777)
        self.seed = 7777
        self.sim_timestep = 1.0 / 60.0
        self.sim_time = 0.0

        self.max_particles = 100_000
        self.sim_particles = 30
        self.particle_radius = 0.1
        self.positions = wp.empty(self.sim_particles, dtype=wp.vec3)
        self.velocities = wp.empty(self.sim_particles, dtype=wp.vec3)
        self.on_mesh = wp.full(self.sim_particles, -1, dtype=wp.int8)
        self.positions_on_mesh = wp.empty(self.max_particles, dtype=wp.vec3)
        self.num_particles_on_mesh = wp.array([0], dtype=wp.int32)

        self.current_particles = 0
        self.emission_counter = 0.0
        self.emission_rate = 1_000  # particles per second
        self.emission_pos = wp.vec3(0.0, 20.0, 0.0)
        self.emission_spread = 0.1
        self.emission_vel = wp.vec3(0.0, 0.0, 0.0)
        self.emission_vel_spread = 1.0

        # create collision mesh
        usd_stage = Usd.Stage.Open(
            os.path.join(warp.examples.get_asset_directory(), "bunny.usd")
        )
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        usd_scale = 10.0
        self.mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get() * usd_scale, dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    # def emit_new_particles(self):
    #     """Emit new particles based on emission rate"""
    #     if self.current_particles >= self.max_particles:
    #         return
    #     self.emission_counter += self.emission_rate * self.sim_timestep

    #     particles_to_emit = int(self.emission_counter)
    #     self.emission_counter -= particles_to_emit  # account for the remainder

    #     wp.launch(
    #         kernel=emit_new_particles,
    #         dim=self.sim_particles,
    #         inputs=[
    #             self.positions,
    #             self.velocities,
    #             self.on_mesh,
    #             self.emission_pos,
    #             self.emission_vel,
    #             self.seed,
    #             self.emission_spread,
    #             self.emission_vel_spread,
    #             self.current_particles,
    #             self.max_particles,
    #         ],
    #     )
    #     self.current_particles += particles_to_emit

    def step(self):
        logging.info(f"Stepping at {self.sim_time}")
        logging.info(f"Postions before emitting:\n{self.positions.numpy()}")
        with wp.ScopedTimer("step"):
            wp.launch(
                kernel=emit_new_particles,
                dim=self.sim_particles,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.on_mesh,
                    self.emission_pos,
                    self.emission_vel,
                    self.seed,
                    self.emission_spread,
                    self.emission_vel_spread,
                ],
            )
            logging.info(f"Positions: after emitting:\n{self.positions.numpy()}")

            wp.launch(
                kernel=simulate,
                dim=self.sim_particles,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.sim_timestep,
                    self.mesh.id,
                    self.on_mesh,
                    self.positions_on_mesh,
                    self.num_particles_on_mesh,
                ],
            )
            logging.info(f"Positions after simulation:\n{self.positions.numpy()}")
            logging.info(f"Positions on mesh:\n{self.positions_on_mesh.numpy()[:30]}")
            logging.info(
                f"Number of particles on mesh:{self.num_particles_on_mesh.numpy()[0]}"
            )
            logging.info(f"On mesh flag:{self.on_mesh.numpy()[:30]}")
            self.sim_time += self.sim_timestep

    def render(self):
        if self.renderer is None:
            return
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_mesh(
                name="mesh",
                points=self.mesh.points.numpy(),
                indices=self.mesh.indices.numpy(),
                colors=(0.35, 0.55, 0.9),
            )
            num_particles_on_mesh = self.num_particles_on_mesh.numpy()[0]
            on_mesh_np = self.on_mesh.numpy()
            active_mask = on_mesh_np == 0
            active_positions = self.positions.numpy()[active_mask]
            # TODO: render poistions_on_mesh

            # points2render = np.vstack(
            #     [
            #         active_positions,
            #         self.positions_on_mesh.numpy()[:num_particles_on_mesh],
            #     ],
            #     dtype=active_positions.dtype,
            # )
            logging.critical(f"Active mask:\n{active_mask}")
            logging.critical(
                f"Positions on mesh to render:\n{self.positions_on_mesh.numpy()[:num_particles_on_mesh]}"
            )
            logging.critical(f"Active positions:\n{active_positions[:30]}")
            # logging.critical(
            #     f"{points2render.shape[0]} points to render:\n{points2render[:30]}\n"
            # )
            self.renderer.render_points(
                name="points",
                points=active_positions,
                radius=self.particle_radius,
                colors=(0.8, 0.3, 0.2),
                # as_spheres=False,
            )
            if num_particles_on_mesh > 0:
                self.renderer.render_points(
                    name="points_on_mesh",
                    points=self.positions_on_mesh.numpy()[:num_particles_on_mesh],
                    radius=self.particle_radius,
                    colors=(0.5, 0.3, 0.8),
                    # as_spheres=False,
                )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override the default Warp device."
    )
    parser.add_argument("--num_frames", type=int, default=500)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        e = Emission()
        for _ in range(args.num_frames):
            e.step()
            e.render()

        if e.renderer:
            e.renderer.save()
