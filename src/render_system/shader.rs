
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
            #extension GL_EXT_ray_query: enable

            layout(location = 0) in vec2 in_uv;
            layout(location = 0) out vec4 f_color;
            
            layout(set = 0, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;

            layout(push_constant) uniform Camera {
                vec3 eye;
                vec3 front;
                vec3 up;
                vec3 right;
                float aspect;
            } camera;

            void main() {
                float t_min = 0.01;
                float t_max = 1000.0;

                vec2 uv = in_uv;

                // ray origin
                vec3 origin = camera.eye;
                
                // ray direction
                vec3 direction = normalize(uv.x * camera.right * camera.aspect + uv.y * camera.up + camera.front);

                rayQueryEXT ray_query;
                rayQueryInitializeEXT(
                    ray_query,
                    top_level_acceleration_structure,
                    gl_RayFlagsCullBackFacingTrianglesEXT,
                    0xFF,
                    origin,
                    t_min,
                    direction,
                    t_max
                );

                // trace ray
                while (rayQueryProceedEXT(ray_query));

                if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                    // miss
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {
                    // hit
                    f_color = vec4(rayQueryGetIntersectionBarycentricsEXT(ray_query, true), 0.0, 1.0);
                }
            }
        ",
    }
}