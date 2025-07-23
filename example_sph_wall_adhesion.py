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
# Example Smoothed Particle Hydrodynamics with Wall Adhesion
#
# Shows how to implement a SPH fluid simulation with particles that can
# stick to walls through adhesion forces.
#
# Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
# Reference Publication
# Matthias Müller, David Charypar, and Markus H. Gross.
# "Particle-based fluid simulation for interactive applications."
# Symposium on Computer animation. Vol. 2. 2003.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
# xyz: relative distances along the axis
# smoothing_length: how far a particle's influence extends
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)
    # return 0 when outside of the smoothing range


@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # only care about happened inside the smoothing range
    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance    # unit vec, from neighbor to current particle
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(smoothing_length - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()


@wp.func
def diff_viscous_kernel(xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()


@wp.func
def wall_adhesion_force(
    position: wp.vec3,
    velocity: wp.vec3,
    adhesion_strength: float,
    adhesion_range: float,
    damping_factor: float,
    width: float,
    height: float,
    length: float
):
    """Calculate adhesion forces to walls within adhesion_range distance"""
    force = wp.vec3()

    # Left wall (x = 0)
    if position[0] < adhesion_range:
        wall_distance = position[0]
        wall_normal = wp.vec3(1.0, 0.0, 0.0)
        strength = adhesion_strength * (adhesion_range - wall_distance) / adhesion_range
        # Add velocity damping towards wall
        velocity_component = wp.dot(velocity, wall_normal)
        damping = -velocity_component * damping_factor if velocity_component < 0.0 else 0.0
        force += wall_normal * (strength + damping)

    # Right wall (x = width)
    if position[0] > width - adhesion_range:
        wall_distance = width - position[0]
        wall_normal = wp.vec3(-1.0, 0.0, 0.0)
        strength = adhesion_strength * (adhesion_range - wall_distance) / adhesion_range
        velocity_component = wp.dot(velocity, wall_normal)
        damping = -velocity_component * damping_factor if velocity_component < 0.0 else 0.0
        force += wall_normal * (strength + damping)

    # Bottom wall (y = 0) - often where you want strongest adhesion
    if position[1] < adhesion_range:
        wall_distance = position[1]
        wall_normal = wp.vec3(0.0, 1.0, 0.0)
        strength = adhesion_strength * (adhesion_range - wall_distance) / adhesion_range
        velocity_component = wp.dot(velocity, wall_normal)
        damping = -velocity_component * damping_factor if velocity_component < 0.0 else 0.0
        force += wall_normal * (strength + damping)

    # Front wall (z = 0)
    if position[2] < adhesion_range:
        wall_distance = position[2]
        wall_normal = wp.vec3(0.0, 0.0, 1.0)
        strength = adhesion_strength * (adhesion_range - wall_distance) / adhesion_range
        velocity_component = wp.dot(velocity, wall_normal)
        damping = -velocity_component * damping_factor if velocity_component < 0.0 else 0.0
        force += wall_normal * (strength + damping)

    # Back wall (z = length)
    if position[2] > length - adhesion_range:
        wall_distance = length - position[2]
        wall_normal = wp.vec3(0.0, 0.0, -1.0)
        strength = adhesion_strength * (adhesion_range - wall_distance) / adhesion_range
        velocity_component = wp.dot(velocity, wall_normal)
        damping = -velocity_component * damping_factor if velocity_component < 0.0 else 0.0
        force += wall_normal * (strength + damping)

    return force


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # store density
    rho = float(0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute density
    # compute the density of a point within the neighborhood
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, smoothing_length)

    # add external potential
    particle_rho[i] = density_normalization * rho


@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_a: wp.array(dtype=wp.vec3),
    isotropic_exp: float,
    base_density: float,
    gravity: float,
    pressure_normalization: float,
    viscous_normalization: float,
    wall_adhesion_strength: float,
    wall_adhesion_range: float,
    wall_damping_factor: float,
    smoothing_length: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute acceleration
    for index in neighbors:
        if index != i:
            # get neighbor velocity
            neighbor_v = particle_v[index]

            # get neighbor density and pressures
            neighbor_rho = particle_rho[index]
            neighbor_pressure = isotropic_exp * (neighbor_rho - base_density)

            # compute relative position
            relative_position = particle_x[index] - x

            # calculate pressure force
            pressure_force += diff_pressure_kernel(
                relative_position, pressure, neighbor_pressure, neighbor_rho, smoothing_length
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)

    # Calculate wall adhesion force
    wall_force = wall_adhesion_force(
        x, v, wall_adhesion_strength, wall_adhesion_range, wall_damping_factor, width, height, length
    )

    # sum all forces
    force = pressure_normalization * pressure_force + viscous_normalization * viscous_force + wall_force

    # add external potential
    particle_a[i] = force / rho + wp.vec3(0.0, gravity, 0.0)


@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    damping_coef: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]

    # clamp x left
    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > width:
        x = wp.vec3(width, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp y bot
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > length:
        x = wp.vec3(x[0], x[1], length)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v


@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), smoothing_length: float, width: float, height: float, length: float
):
    # one tid points to one particle
    # tid in [0, num_particle-1]
    tid = wp.tid()

    # grid size
    # num_particle along axis
    nr_x = wp.int32(width / 4.0 / smoothing_length)     # width
    nr_y = wp.int32(height / smoothing_length)          # height
    nr_z = wp.int32(length / 4.0 / smoothing_length)    # length

    # calculate particle position(tid -> xyz-coordinate)
    # full the particle first along z-axis
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos


class Example:
    def __init__(self, stage_path="example_sph_wall_adhesion.usd", verbose=False):
        self.verbose = verbose

        # render params
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0

        # simulation params
        self.smoothing_length = 0.8  # NOTE change this to adjust number of particles
        self.width = 80.0  # x
        self.height = 80.0  # y
        self.length = 80.0  # z
        self.isotropic_exp = 20
        self.base_density = 1.0
        self.particle_mass = 0.01 * self.smoothing_length**3  # reduce according to smoothing length
        self.dt = 0.01 * self.smoothing_length  # decrease sim dt by smoothing length
        self.dynamic_visc = 0.025
        self.damping_coef = -0.95
        self.gravity = -0.1
        self.n = int(
            self.height * (self.width / 4.0) * (self.height / 4.0) / (self.smoothing_length**3)
        )  # number particles (small box in corner)
        self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)

        # Wall adhesion parameters
        self.wall_adhesion_strength = 0.1  # Adjust this to control stickiness (start with 0.01-0.1)
        self.wall_adhesion_range = self.smoothing_length * 3.0  # Range from wall where adhesion applies
        self.wall_damping_factor = 2.0  # Velocity damping near walls

        # constants
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )  # integrate density kernel
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.smoothing_length**6)
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )

        # allocate arrays
        self.x = wp.empty(self.n, dtype=wp.vec3)
        self.v = wp.zeros(self.n, dtype=wp.vec3)
        self.rho = wp.zeros(self.n, dtype=float)
        self.a = wp.zeros(self.n, dtype=wp.vec3)

        # set random positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[self.x, self.smoothing_length, self.width, self.height, self.length],
        )  # initialize in small area

        # create hash array
        grid_size = int(self.height / (4.0 * self.smoothing_length))
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

        # renderer
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)

    def step(self):
        with wp.ScopedTimer("step"):
            for _ in range(self.sim_step_to_frame_ratio):
                with wp.ScopedTimer("grid build", active=self.verbose):

                    # build grid
                    # smoothing_length is the interaction range of a particle
                    self.grid.build(self.x, self.smoothing_length)

                with wp.ScopedTimer("forces", active=self.verbose):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.smoothing_length],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.rho,
                            self.a,
                            self.isotropic_exp,
                            self.base_density,
                            self.gravity,
                            self.pressure_normalization,
                            self.viscous_normalization,
                            self.wall_adhesion_strength,
                            self.wall_adhesion_range,
                            self.wall_damping_factor,
                            self.smoothing_length,
                            self.width,
                            self.height,
                            self.length,
                        ],
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.width, self.height, self.length],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt])

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_points(
                points=self.x.numpy(), radius=self.smoothing_length, name="points", colors=(0.8, 0.3, 0.2)
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_sph_wall_adhesion.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=480, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        for _ in range(args.num_frames):
            example.render()
            example.step()

        if example.renderer:
            example.renderer.save()