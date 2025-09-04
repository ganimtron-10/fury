{{ bindings_code }}
{$ include 'fury.utils.wgsl' $}

const NUM_LINES = i32({{ num_lines }});

@compute @workgroup_size({{ workgroup_size }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let line_id = i32(global_id.x);

    let line_start = load_s_offsets(line_id) + line_id;
    let line_end = line_start + load_s_lengths(line_id);
    var intersect = -1.0;
    var p0 = vec3<f32>(0.0, 0.0, 0.0);
    var p1 = vec3<f32>(0.0, 0.0, 0.0);
    for (var i: i32 = line_start + 1; i < line_end; i++) {
        p0 = load_s_lines(i - 1);
        p1 = load_s_lines(i);
        let plane = u_wobject.plane;

        let t = crosssect_plane(plane, p0, p1);
        if (0.0 <= t && t <= 1.0) {
            intersect = t;
            break;
        }
    }

    if (intersect != -1.0) {
        let intersection = intersect_plane(intersect, p0, p1);
        let point = perpendicular_point(intersection, vec3<f32>(u_wobject.plane.xyz), u_wobject.lift);
        s_positions[line_id * 3] = point.x;
        s_positions[line_id * 3 + 1] = point.y;
        s_positions[line_id * 3 + 2] = point.z;
        s_colors[line_id] = vec4<f32>(s_colors[line_id].xyz, 1.0);
        s_edge_colors[line_id] = vec4<f32>(s_edge_colors[line_id].xyz, 1.0);
    } else {
        s_colors[line_id] = vec4<f32>(s_colors[line_id].xyz, 0.0);
        s_edge_colors[line_id] = vec4<f32>(s_edge_colors[line_id].xyz, 0.0);
    }
}
