vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, scalar) readonly buffer InputOrigin {
    vec3 input_origin[];
};

layout(set = 0, binding = 1, scalar) readonly buffer InputsDirection {
    vec3 input_direction[];
};

layout(set = 0, binding = 2, scalar) readonly buffer InputsEmissivity {
    vec3 input_emissivity[];
};

layout(set = 0, binding = 3, scalar) readonly buffer InputsReflectivity {
    vec3 input_reflectivity[];
};

layout(set = 0, binding = 4) readonly buffer InputsNeeMisWeight {
    float input_nee_mis_weight[];
};

layout(set = 0, binding = 5) readonly buffer InputsBsdfPdf {
    float input_bsdf_pdf[];
};

layout(set = 0, binding = 6) readonly buffer InputsNeePdf {
    float input_nee_pdf[];
};

layout(set = 0, binding = 7, scalar) readonly buffer InputsDebugInfo {
    vec4 input_debug_info[];
};

layout(set = 0, binding = 8) writeonly buffer OutputsImage {
    u8vec4 output_image[];
};

layout(push_constant, scalar) uniform PushConstants {
    uint num_bounces;
    uint num_samples;
    uint xsize;
    uint ysize;
};

void main() {
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }

    vec3 color = vec3(0.0);
    for (uint sample_id = 0; sample_id < num_samples; sample_id++) {
        // compute the color for this sample
        vec3 sample_color = vec3(0.0);
        for(int bounce = int(num_bounces)-1; bounce >= 0; bounce--) {            
            // tensor layout: [bounce, sample, y, x, channel]
            const uint bid = bounce         * num_samples * ysize * xsize 
                + gl_GlobalInvocationID.z   * ysize * xsize 
                + gl_GlobalInvocationID.y   * xsize 
                + gl_GlobalInvocationID.x;
            // whether the ray is valid
            float ray_valid = input_direction[bid] == vec3(0.0) ? 0.0 : 1.0;

            // compute importance sampling data
            float bsdf_pdf = input_bsdf_pdf[bid];
            float nee_pdf = input_nee_pdf[bid];
            float nee_mis_weight = input_nee_mis_weight[bid];
            // this is our sampling distribution: 
            // mis_weight proportion of the time, we sample from the light source, and 1-mis_weight proportion of the time, we sample from the BSDF
            float qx = nee_pdf * nee_mis_weight + (1.0 - nee_mis_weight) * bsdf_pdf;
            float reweighting_factor = bsdf_pdf / qx;

            sample_color = input_emissivity[bid] + sample_color * input_reflectivity[bid] * reweighting_factor * ray_valid;
        }
        color += sample_color;
    }

    // average the samples
    vec3 pixel_color = color / float(num_samples);
    output_image[gl_GlobalInvocationID.y*xsize + gl_GlobalInvocationID.x] = u8vec4(pixel_color.zyx*255, 255);
}
",
}
