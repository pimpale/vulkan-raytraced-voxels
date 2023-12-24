
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 position;
            layout(location = 0) out vec2 out_uv;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                out_uv = position;
            }
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460
            #extension GL_EXT_ray_query : enable

            layout(location = 0) in vec2 in_uv;
            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;

            void main() {
                float t_min = 0.01;
                float t_max = 1000.0;
                vec3 origin = vec3(0.0, 0.0, 0.0);
                vec3 direction = normalize(vec3(in_uv * 1.0, 1.0));

                rayQueryEXT ray_query;
                rayQueryInitializeEXT(
                    ray_query,
                    top_level_acceleration_structure,
                    gl_RayFlagsTerminateOnFirstHitEXT,
                    0xFF,
                    origin,
                    t_min,
                    direction,
                    t_max
                );
                rayQueryProceedEXT(ray_query);

                if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                    // miss
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    // hit
                    f_color = vec4(1.0, 0.0, 0.0, 1.0);
                }
            }
        ",
    }
}