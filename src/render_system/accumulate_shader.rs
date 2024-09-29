vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 1, binding = 4) readonly buffer InputOrigin {
    vec3 input_origin[];
};

layout(set = 1, binding = 5) readonly buffer InputsDirection {
    vec3 input_direction[];
};

layout(set = 1, binding = 6) readonly buffer InputsEmissivity {
    vec3 input_emissivity[];
};

layout(set = 1, binding = 7) readonly buffer InputsReflectivity {
    vec3 input_reflectivity[];
};

layout(set = 1, binding = 8) readonly buffer InputsRayPdfOverScatterPdf {
    float input_ray_pdf_over_scatter_pdf[];
};

layout(set = 1, binding = 9) writeonly buffer InputsDebugInfo {
    vec4 input_debug_info[];
};

struct Camera {
    vec3 eye;
    vec3 front;
    vec3 up;
    vec3 right;
    uvec2 screen_size;
};

layout(push_constant, scalar) uniform PushConstants {
    Camera camera;
    uint frame;
} push_constants;

void main() {
    Camera camera = push_constants.camera;
    if(gl_GlobalInvocationID.x >= camera.screen_size.x || gl_GlobalInvocationID.y >= camera.screen_size.y) {
        return;
    }


    vec3 color = vec3(0.0);
    for (uint sample_id = 0; sample_id < SAMPLES_PER_PIXEL; sample_id++) {

        
        // compute the color for this sample
        vec3 sample_color = vec3(0.0);
        for(int i = int(current_bounce)-1; i >= 0; i--) {
            sample_color = bounce_emissivity[i] + sample_color * bounce_reflectivity[i];
        }
        if (current_bounce > 1 && push_constants.frame % 100 > 50) {
            sample_color = bounce_debuginfo[0];
        }
        color += sample_color;
    }


    // average the samples
    vec3 pixel_color = (1.0*color) / float(SAMPLES_PER_PIXEL);
    out_color[gl_GlobalInvocationID.y*camera.screen_size.x + gl_GlobalInvocationID.x] = u8vec4(pixel_color.zyx*255, 255);
}
",
}
