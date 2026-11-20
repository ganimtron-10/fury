// Sphere physics on the GPU: position-based dynamics with a hashed grid.
//
// One substep runs: predict, clear grid, insert, (solve, apply) x iterations,
// finalize. Every kernel handles one sphere per invocation and writes only
// that sphere's records, except insert, which adds spheres to shared grid
// cells through atomics. No kernel reads a record that it also writes.
//
// Static spheres much larger than the moving ones are kept out of the grid,
// whose cells are sized for the moving spheres, and checked directly as
// obstacles instead.
//
// Per-sphere scratch records during a substep:
//   s_previous: xyz position at the start of the substep, w the total length
//     the sphere was pushed by other spheres (used for friction between them).
//   s_contacts: xyz the total contact push, w the fastest speed anything
//     pushing the sphere moved towards it.
//   s_corrected: xyz the corrected position from solve, w the length of this
//     iteration's push from other spheres; apply adds it to s_previous.w.
//   s_surroundings: xyz how far touching spheres have moved this substep,
//     weighted by overlap, w the sum of the weights. Friction between spheres
//     acts on velocity relative to their average motion. It uses motion, not
//     stored velocities, so velocities cannot feed on each other. Each
//     iteration samples the neighbours before that iteration's corrections,
//     so the reference lags by one iteration; more iterations shrink the lag.

struct Params {
    gravity_dt: vec4<f32>,
    grid: vec4<f32>,
    counts: vec4<u32>,
    box_center: vec4<f32>,
    box_half: vec4<f32>,
    box_axis_x: vec4<f32>,
    box_axis_y: vec4<f32>,
    box_axis_z: vec4<f32>,
    ground: vec4<f32>,
    misc: vec4<f32>,
    box_linear: vec4<f32>,
    box_angular: vec4<f32>,
};

// kind: 1 uniform force, 2 relaxation (drag, wind), 3 attractor, 4 vortex,
// 5 one-shot blast, applied by the blast kernel before a step.
struct Force {
    kind_values: vec4<f32>,
    a: vec4<f32>,
    b: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_params: Params;
@group(0) @binding(1) var<storage, read_write> s_positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> s_velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> s_previous: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> s_corrected: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> s_contacts: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> s_cell_counts: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> s_cell_items: array<u32>;
@group(0) @binding(8) var<storage, read_write> s_stats: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> s_forces: array<Force>;
@group(0) @binding(10) var<storage, read> s_obstacles: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> s_surroundings: array<vec4<f32>>;

const WORKGROUP_SIZE: u32 = {{ workgroup_size }}u;

// Large worlds need more workgroups than one dispatch dimension allows, so
// kernels are dispatched as rows of workgroups.
fn item_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * WORKGROUP_SIZE;
}

fn sphere_count() -> u32 {
    return u_params.counts.x;
}

fn cell_coord(position: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(position * u_params.grid.y));
}

fn cell_hash(coord: vec3<i32>) -> u32 {
    let hashed = (bitcast<u32>(coord.x) * 73856093u) ^
        (bitcast<u32>(coord.y) * 19349663u) ^
        (bitcast<u32>(coord.z) * 83492791u);
    return hashed & u_params.counts.y;
}

fn force_acceleration(position: vec3<f32>, inverse_mass: f32) -> vec3<f32> {
    var acceleration = u_params.gravity_dt.xyz;
    let force_count = u_params.counts.w;
    for (var index = 0u; index < force_count; index += 1u) {
        let force = s_forces[index];
        let kind = u32(force.kind_values.x);
        if (kind == 1u) {
            acceleration += force.a.xyz * inverse_mass;
        } else if (kind == 3u) {
            let strength = force.kind_values.y;
            let softening = force.kind_values.z;
            let offset = force.a.xyz - position;
            let distance2 = dot(offset, offset) + softening * softening;
            acceleration += strength * offset / pow(distance2, 1.5);
        } else if (kind == 4u) {
            let strength = force.kind_values.y;
            let pull = force.kind_values.z;
            let axis = force.b.xyz;
            let offset = position - force.a.xyz;
            let radial = offset - axis * dot(offset, axis);
            let distance2 = dot(radial, radial) + 1e-3;
            acceleration += cross(axis, radial) * strength / distance2;
            acceleration -= radial * pull / sqrt(distance2);
        }
    }
    return acceleration;
}

fn relax_velocity(velocity: vec3<f32>, inverse_mass: f32, dt: f32) -> vec3<f32> {
    var total = 0.0;
    var target_sum = vec3<f32>(0.0);
    let force_count = u_params.counts.w;
    for (var index = 0u; index < force_count; index += 1u) {
        let force = s_forces[index];
        if (u32(force.kind_values.x) == 2u) {
            total += force.kind_values.w;
            target_sum += force.kind_values.w * force.a.xyz;
        }
    }
    if (total <= 0.0) {
        return velocity;
    }
    let target_velocity = target_sum / total;
    return target_velocity + (velocity - target_velocity) * exp(-total * inverse_mass * dt);
}

fn blast_velocity(position: vec3<f32>) -> vec3<f32> {
    var gain = vec3<f32>(0.0);
    let force_count = u_params.counts.w;
    for (var index = 0u; index < force_count; index += 1u) {
        let force = s_forces[index];
        if (u32(force.kind_values.x) != 5u) {
            continue;
        }
        let strength = force.kind_values.y;
        let radius = force.kind_values.z;
        let offset = position - force.a.xyz;
        let distance = length(offset);
        if (distance >= radius) {
            continue;
        }
        var direction = vec3<f32>(0.0, 1.0, 0.0);
        if (distance > 1e-9) {
            direction = offset / distance;
        }
        gain += direction * strength * (1.0 - distance / radius);
    }
    return gain;
}

fn clamp_speed(velocity: vec3<f32>) -> vec3<f32> {
    let max_speed = u_params.grid.w;
    let speed = length(velocity);
    if (speed > max_speed) {
        return velocity * (max_speed / speed);
    }
    return velocity;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn clear(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let cell = item_index(gid, groups);
    if (cell <= u_params.counts.y) {
        atomicStore(&s_cell_counts[cell], 0u);
    }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn predict(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = item_index(gid, groups);
    if (index >= sphere_count()) {
        return;
    }
    var position = s_positions[index];
    var velocity = s_velocities[index];
    s_previous[index] = vec4<f32>(position.xyz, 0.0);
    s_contacts[index] = vec4<f32>(0.0);
    s_surroundings[index] = vec4<f32>(0.0);
    let inverse_mass = velocity.w;
    if (inverse_mass <= 0.0) {
        return;
    }
    let dt = u_params.gravity_dt.w;
    var linear = velocity.xyz;
    linear += force_acceleration(position.xyz, inverse_mass) * dt;
    linear = clamp_speed(relax_velocity(linear, inverse_mass, dt));
    position = vec4<f32>(position.xyz + linear * dt, position.w);
    s_positions[index] = position;
    s_velocities[index] = vec4<f32>(linear, inverse_mass);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn blast(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = item_index(gid, groups);
    if (index >= sphere_count()) {
        return;
    }
    let velocity = s_velocities[index];
    if (velocity.w <= 0.0) {
        return;
    }
    let gain = blast_velocity(s_positions[index].xyz);
    s_velocities[index] = vec4<f32>(velocity.xyz + gain, velocity.w);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn insert(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = item_index(gid, groups);
    if (index >= sphere_count()) {
        return;
    }
    let sphere = s_positions[index];
    if (is_obstacle(sphere, s_velocities[index].w)) {
        return;
    }
    let bucket = u_params.counts.z;
    let cell = cell_hash(cell_coord(sphere.xyz));
    let slot = atomicAdd(&s_cell_counts[cell], 1u);
    if (slot < bucket) {
        s_cell_items[cell * bucket + slot] = index;
    } else {
        atomicAdd(&s_stats[0], 1u);
    }
}

fn is_obstacle(sphere: vec4<f32>, inverse_mass: f32) -> bool {
    return inverse_mass <= 0.0 && 2.0 * sphere.w > u_params.grid.x * 1.0001;
}

fn obstacle_count() -> u32 {
    return u32(u_params.box_linear.w);
}

fn box_local(position: vec3<f32>) -> vec3<f32> {
    let relative = position - u_params.box_center.xyz;
    return vec3<f32>(
        dot(relative, u_params.box_axis_x.xyz),
        dot(relative, u_params.box_axis_y.xyz),
        dot(relative, u_params.box_axis_z.xyz)
    );
}

// The walls hold only spheres whose centre was inside the box at the start of
// the substep, as in the CPU Container: spheres outside are not pulled in, and
// spheres inside cannot tunnel out however fast they move. The box pose is the
// one at the end of the substep, so the start point is carried along with the
// walls before the test.
fn box_correction(position: vec3<f32>, start: vec3<f32>, radius: f32) -> vec3<f32> {
    if (u_params.box_center.w == 0.0) {
        return vec3<f32>(0.0);
    }
    let carried = start + wall_velocity(start) * u_params.gravity_dt.w;
    if (any(abs(box_local(carried)) > u_params.box_half.xyz)) {
        return vec3<f32>(0.0);
    }
    let axis_x = u_params.box_axis_x.xyz;
    let axis_y = u_params.box_axis_y.xyz;
    let axis_z = u_params.box_axis_z.xyz;
    let local_position = box_local(position);
    let open_top = u_params.box_half.w > 0.0;
    if (open_top && local_position.y > u_params.box_half.y) {
        // Above the rim of an open box nothing holds the sphere.
        return vec3<f32>(0.0);
    }
    let limit = max(u_params.box_half.xyz - vec3<f32>(radius), vec3<f32>(0.0));
    var clamped = clamp(local_position, -limit, limit);
    if (open_top) {
        clamped.y = max(local_position.y, -limit.y);
    }
    let push = clamped - local_position;
    return axis_x * push.x + axis_y * push.y + axis_z * push.z;
}

fn wall_velocity(position: vec3<f32>) -> vec3<f32> {
    let arm = position - u_params.box_center.xyz;
    return u_params.box_linear.xyz + cross(u_params.box_angular.xyz, arm);
}

// Positional Coulomb friction against the walls: while a wall pushes a sphere
// by `push`, the sphere may not slide along it by more than friction times
// that push in a substep. Without it, pile pressure squeezes spheres up the
// walls and into towers in the corners.
fn wall_friction(position: vec3<f32>, start: vec3<f32>, push: vec3<f32>) -> vec3<f32> {
    let push_length = length(push);
    if (push_length <= 0.0) {
        return position;
    }
    let normal = push / push_length;
    let moved = position - start;
    let sliding = moved - normal * dot(moved, normal);
    let sliding_length = length(sliding);
    let limit = u_params.grid.z * push_length;
    if (sliding_length <= limit) {
        return position - sliding;
    }
    return position - sliding * (limit / sliding_length);
}

fn ground_correction(position: vec3<f32>, radius: f32) -> vec3<f32> {
    if (u_params.misc.x == 0.0) {
        return vec3<f32>(0.0);
    }
    let normal = u_params.ground.xyz;
    let depth = dot(normal, position) - u_params.ground.w - radius;
    if (depth >= 0.0) {
        return vec3<f32>(0.0);
    }
    return -depth * normal;
}

// Direction pushing `index` away from `other`. Coincident spheres get a fixed
// direction by index order so they still separate.
fn contact_normal(offset: vec3<f32>, distance: f32, index: u32, other: u32) -> vec3<f32> {
    if (distance > 1e-6) {
        return offset / distance;
    }
    return vec3<f32>(select(-1.0, 1.0, index > other), 0.0, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn solve(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = item_index(gid, groups);
    if (index >= sphere_count()) {
        return;
    }
    let sphere = s_positions[index];
    let inverse_mass = s_velocities[index].w;
    if (inverse_mass <= 0.0) {
        s_corrected[index] = vec4<f32>(sphere.xyz, 0.0);
        return;
    }
    let bucket = u_params.counts.z;
    let base = cell_coord(sphere.xyz);
    var shift = vec3<f32>(0.0);
    var touching = 0u;
    // Fastest speed something pushing this sphere moves towards it. Contacts
    // may separate the sphere no faster than that, so pressure inside a pile
    // cannot launch spheres.
    var separation = 0.0;
    var surroundings = vec4<f32>(0.0);
    for (var dz = -1; dz <= 1; dz += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            for (var dx = -1; dx <= 1; dx += 1) {
                let cell = cell_hash(base + vec3<i32>(dx, dy, dz));
                let filled = min(atomicLoad(&s_cell_counts[cell]), bucket);
                for (var slot = 0u; slot < filled; slot += 1u) {
                    let other = s_cell_items[cell * bucket + slot];
                    if (other == index) {
                        continue;
                    }
                    let neighbour = s_positions[other];
                    let offset = sphere.xyz - neighbour.xyz;
                    let reach = sphere.w + neighbour.w;
                    let distance2 = dot(offset, offset);
                    if (distance2 >= reach * reach) {
                        continue;
                    }
                    let distance = sqrt(distance2);
                    let other_inverse_mass = s_velocities[other].w;
                    let share = inverse_mass / (inverse_mass + other_inverse_mass);
                    let normal = contact_normal(offset, distance, index, other);
                    shift += normal * (reach - distance) * share;
                    touching += 1u;
                    let weight = reach - distance;
                    let moved = neighbour.xyz - s_previous[other].xyz;
                    surroundings += vec4<f32>(moved * weight, weight);
                    let approach = dot(s_velocities[other].xyz, normal);
                    separation = max(separation, approach);
                }
            }
        }
    }
    let obstacles = obstacle_count();
    for (var obstacle = 0u; obstacle < obstacles; obstacle += 1u) {
        let blocker = s_obstacles[obstacle];
        let offset = sphere.xyz - blocker.xyz;
        let reach = sphere.w + blocker.w;
        let distance2 = dot(offset, offset);
        if (distance2 >= reach * reach) {
            continue;
        }
        let distance = sqrt(distance2);
        shift += contact_normal(offset, distance, 1u, 0u) * (reach - distance);
        touching += 1u;
        surroundings.w += reach - distance;
    }
    var corrected = sphere.xyz;
    var sphere_push = vec3<f32>(0.0);
    if (touching > 0u) {
        sphere_push = shift * (u_params.misc.y / f32(touching));
        corrected += sphere_push;
    }
    let start = s_previous[index];
    let box_push = box_correction(corrected, start.xyz, sphere.w);
    var wall_motion = vec3<f32>(0.0);
    if (dot(box_push, box_push) > 0.0) {
        wall_motion = wall_velocity(start.xyz) * u_params.gravity_dt.w;
        let wall_speed = dot(wall_velocity(corrected), normalize(box_push));
        separation = max(separation, wall_speed);
    }
    corrected += box_push;
    let ground_push = ground_correction(corrected, sphere.w);
    corrected += ground_push;
    corrected = wall_friction(corrected, start.xyz + wall_motion, box_push + ground_push);
    let contact = s_contacts[index];
    s_contacts[index] = vec4<f32>(
        contact.xyz + sphere_push + box_push + ground_push, max(contact.w, separation)
    );
    s_surroundings[index] += surroundings;
    s_corrected[index] = vec4<f32>(corrected, length(sphere_push));
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = item_index(gid, groups);
    if (index >= sphere_count()) {
        return;
    }
    let sphere = s_positions[index];
    let corrected = s_corrected[index];
    s_positions[index] = vec4<f32>(corrected.xyz, sphere.w);
    let start = s_previous[index];
    s_previous[index] = vec4<f32>(start.xyz, start.w + corrected.w);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn finalize(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let index = item_index(gid, groups);
    if (index >= sphere_count()) {
        return;
    }
    let state = s_velocities[index];
    if (state.w <= 0.0) {
        return;
    }
    let dt = u_params.gravity_dt.w;
    let predicted = state.xyz;
    let start = s_previous[index];
    var velocity = (s_positions[index].xyz - start.xyz) / dt;
    let contact = s_contacts[index];
    let push = contact.xyz;
    let push_length = length(push);
    if (push_length > 1e-7) {
        let normal = push / push_length;
        let normal_speed = dot(velocity, normal);
        let tangent = velocity - normal * normal_speed;
        let bounce = max(-u_params.misc.z * dot(predicted, normal), 0.0);
        let leaving = max(min(normal_speed, max(contact.w, bounce)), bounce);
        // Coulomb friction between spheres: the grip grows with how hard other
        // spheres pushed, so spheres merely grazing each other keep sliding.
        // Wall friction was already applied to the position in solve.
        let surroundings = s_surroundings[index];
        var neighbour_velocity = vec3<f32>(0.0);
        if (surroundings.w > 0.0) {
            neighbour_velocity = surroundings.xyz / (surroundings.w * dt);
        }
        let relative = velocity - neighbour_velocity;
        let slip = relative - normal * dot(relative, normal);
        let slip_speed = length(slip);
        var sliding = tangent;
        if (slip_speed > 1e-9) {
            let grip = min(slip_speed, u_params.grid.z * start.w / dt);
            sliding = tangent - slip * (grip / slip_speed);
        }
        velocity = sliding + normal * leaving;
    }
    s_velocities[index] = vec4<f32>(clamp_speed(velocity), state.w);
}
