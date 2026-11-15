{$ include 'pygfx.std.wgsl' $}
{$ include 'fury.utils.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var varyings: Varyings;

    // PyGfx handles index buffers manually via storage buffers!
    // in.index is a flat counter (0, 1, 2, 3, 4, 5...)
    let flat_index = i32(in.index);
    let face_index = flat_index / 3;
    let sub_index = flat_index % 3;

    // Load the indices for the current face (triangle)
    let vii = load_s_indices(face_index);
    let i0 = i32(vii[sub_index]);

    // Load position from the storage buffer using the correct vertex index
    let raw_pos = load_s_positions(i0);
    let local_pos = raw_pos.xyz;

    // Transform position to world space
    let world_pos = u_wobject.world_transform * vec4<f32>(local_pos, 1.0);
    // Transform position to clip space
    let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;

    varyings.position = vec4<f32>(clip_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);

    // Pass the local unscaled position (which FURY scaled via plane_geometry width/height)
    varyings.texcoord_vert = vec2<f32>(local_pos.xy);

    return varyings;
}

// Signed Distance Function for a rounded box
// p = current point position (relative to center 0,0)
// b = half-size of the box (width/2, height/2)
// r = corner radius
fn sd_round_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - r;
}

// Pick the radius of the corner in the quadrant of p.
// radii = (top-left, top-right, bottom-right, bottom-left), with +y up.
fn corner_radius_at(p: vec2<f32>, radii: vec4<f32>) -> f32 {
    let right = select(radii.z, radii.y, p.y > 0.0);
    let left = select(radii.w, radii.x, p.y > 0.0);
    return select(left, right, p.x > 0.0);
}

// Corner radii limited to half of the smaller side of the box.
fn clamped_radii(size: vec2<f32>) -> vec4<f32> {
    let max_radius = 0.5 * min(size.x, size.y);
    return min(max(u_material.corner_radius, vec4<f32>(0.0)), vec4<f32>(max_radius));
}

// Abramowitz and Stegun 7.1.27 approximation of the error function.
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 + (0.278393 + (0.230389 + (0.000972 + 0.078108 * a) * a) * a) * a;
    let t2 = t * t;
    return s - s / (t2 * t2);
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let size = u_material.size;
    let radii = clamped_radii(size);

    // The geometry is created with the exact physical width and height
    // (enlarged by the shadow extent for the shadow pass). Therefore,
    // varyings.texcoord_vert already contains the physical pixel offset
    // from the center of the drawn shape!
    let p = varyings.texcoord_vert.xy;
    let radius = corner_radius_at(p, radii);

    $$ if shadow_pass

    // The shadow is the box shape grown by the spread and blurred with a
    // gaussian of sigma = blur / 2. The convolution of a straight edge with a
    // gaussian is an erf of the signed distance, which is a close and cheap
    // approximation for the rounded shape as well.
    let spread = u_material.shadow_spread;
    let b = max(size * 0.5 + vec2<f32>(spread), vec2<f32>(0.0));
    let shadow_radius = select(0.0, max(radius + spread, 0.0), radius > 0.0);
    let distance = sd_round_box(p, b, shadow_radius);
    let smooth_edge = fwidth(distance);

    // A blur below a pixel would be sharper than the anti-aliasing.
    let sigma = select(
        0.0, max(0.5 * u_material.shadow_blur, 0.5 * smooth_edge), u_material.shadow_blur > 0.0
    );
    var coverage = 1.0 - smoothstep(-smooth_edge, smooth_edge, distance);
    if (sigma > 0.0) {
        coverage = 0.5 * (1.0 - erf_approx(distance / (sigma * 1.41421356)));
    }

    // Like a CSS box shadow, the shadow is not drawn under the box itself,
    // so it does not show through a translucent box. The quad is centered
    // on the shadow, so the box sits at -shadow_offset. The cut ends at the
    // box edge, so no light fringe appears just outside an opaque box.
    let box_p = p + u_material.shadow_offset;
    let box_distance = sd_round_box(box_p, size * 0.5, corner_radius_at(box_p, radii));
    let box_edge = fwidth(box_distance);
    coverage *= smoothstep(-box_edge, box_edge, box_distance + box_edge);

    let shadow_color = u_material.shadow_color;
    let final_alpha = shadow_color.a * u_material.opacity * coverage;
    if (final_alpha <= 0.0) {
        discard;
    }

    var out: FragmentOutput;
    out.color = vec4<f32>(srgb2physical(shadow_color.rgb), final_alpha);

    $$ else

    let b = size * 0.5;

    // Evaluate the SDF
    let distance = sd_round_box(p, b, radius);

    // Smooth the edge using fwidth for perfect anti-aliasing
    let smooth_edge = fwidth(distance);
    let alpha = 1.0 - smoothstep(-smooth_edge, smooth_edge, distance);

    // Discard pixels completely outside the rounded corners
    if (alpha <= 0.0) {
        discard;
    }

    // Fetch base color from the material
    let base_color = u_material.color;
    var physical_color = srgb2physical(base_color.rgb);
    var color_alpha = base_color.a;

    // The border is the band between the box edge and the same shape inset
    // by the border width. Insetting the SDF also shrinks the inner corner
    // radii, so the band keeps a constant width around the corners.
    let border_width = u_material.border_width;
    if (border_width > 0.0) {
        let border_mix = smoothstep(-smooth_edge, smooth_edge, distance + border_width);
        let border_color = u_material.border_color;
        // Composite the border over the fill (source-over, premultiplied),
        // then un-premultiply because the blender expects straight alpha.
        // Opacity fades the composited box as a whole, so it is applied
        // once afterwards and an opaque border still hides the fill.
        let border_alpha = border_color.a * border_mix;
        let mixed_alpha = border_alpha + color_alpha * (1.0 - border_alpha);
        let premultiplied = srgb2physical(border_color.rgb) * border_alpha
            + physical_color * color_alpha * (1.0 - border_alpha);
        if (mixed_alpha > 0.0) {
            physical_color = premultiplied / mixed_alpha;
        }
        color_alpha = mixed_alpha;
    }

    // Multiply material opacity by the SDF alpha
    let final_alpha = color_alpha * u_material.opacity * alpha;

    var out: FragmentOutput;
    out.color = vec4<f32>(physical_color, final_alpha);

    $$ endif

    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(0u, 26) +
        pick_pack(0u, 18)
    );
    $$ endif

    return out;
}
