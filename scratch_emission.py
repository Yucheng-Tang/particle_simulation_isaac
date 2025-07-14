import numpy as np
import warp as wp
import warp.render

@wp.kernel
def simulate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    sim_timestep: float
):
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, 0.0 - 9.8, 0.0) * sim_timestep - v * 0.1 * sim_timestep
    xpred = x + v * sim_timestep

    # pbd update
    v = (xpred - x) * (1.0 / sim_timestep)
    x = xpred

    positions[tid] = x
    velocities[tid] = v

class Emission:
    def __init__(self, stage_path='emission.usd'):
        self.rng = np.random.default_rng(7777)
        self.sim_timestep = 1.0 / 60.0
        self.sim_time = 0.0

        self.max_particles = 100_000
        self.particle_radius = 0.01
        self.positions = wp.zeros(self.max_particles, dtype=wp.vec3)
        self.velocities = wp.zeros(self.max_particles, dtype=wp.vec3)

        self.current_particles = 0
        self.emission_counter = 0.0
        self.emission_rate = 10_000     # particles per second
        self.emission_pos = np.array([0.0, 5.0, 0.0])
        self.emission_spread = 0.5
        self.emission_vel = np.zeros(3)
        self.emission_vel_spread = 1.0

        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    def emit_new_particles(self):
        """Emit new particles based on emission rate"""
        self.emission_counter += self.emission_rate * self.sim_timestep

        particles_to_emit = int(self.emission_counter)
        self.emission_counter -= particles_to_emit      # account for the remainder

        for _ in range(particles_to_emit):
            if self.current_particles >= self.max_particles:
                break

            # Random position around emission point
            pos = self.emission_pos + (self.rng.random(3) - 0.5) * self.emission_spread
            vel = self.emission_vel + (self.rng.random(3) - 0.5) * self.emission_vel_spread

            # Add particle to arrays
            self.positions.numpy()[self.current_particles] = pos
            self.velocities.numpy()[self.current_particles] = vel

            self.current_particles += 1

    def step(self):
        with wp.ScopedTimer("step"):
            self.emit_new_particles()
            if self.current_particles > 0:
                wp.launch(
                    kernel = simulate,
                    dim = self.current_particles,
                    inputs = [self.positions, self.velocities, self.sim_timestep]
                )
            self.sim_time += self.sim_timestep

    def render(self):
        if self.render is None:
            return
        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                name="points", points=self.positions.numpy(), radius=self.particle_radius, colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=500)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        e = Emission()
        for _ in range(args.num_frames):
            e.step()
            e.render()

        if e.renderer:
            e.renderer.save()

