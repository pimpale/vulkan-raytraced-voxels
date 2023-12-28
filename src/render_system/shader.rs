
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
            #extension GL_EXT_ray_query: require
            #extension GL_EXT_scalar_block_layout: require
            #extension GL_EXT_buffer_reference2: require
            #extension GL_EXT_shader_explicit_arithmetic_types_int64: require
            #extension GL_EXT_nonuniform_qualifier: require

            layout(location = 0) in vec2 in_uv;
            layout(location = 0) out vec4 f_color;
            
            layout(set = 0, binding = 0) uniform sampler s;
            layout(set = 0, binding = 1) uniform texture2D tex[];

            layout(set = 1, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;
            
            struct Vertex {
                vec3 position;
                uint t;
                vec2 uv;
            };

            layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer InstanceData {
                Vertex vertexes[];
            };

            layout(set = 1, binding = 1) readonly buffer InstanceDataAddresses {
                // one uint64 per instance that points to the device address of the data for that instance
                uint64_t instance_data_addrs[];
            };

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
                    0,
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
                    uint prim_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
                    uint instance_index = rayQueryGetIntersectionInstanceIdEXT(ray_query, true);

                    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);
                    vec3 bary3 = vec3(1.0 - bary.x - bary.y,  bary.x, bary.y);

                    // get the instance data for this instance
                    InstanceData id = InstanceData(instance_data_addrs[instance_index]);

                    // get the texture coordinates
                    uint t = id.vertexes[prim_index*3 + 0].t;
                    vec2 uv = id.vertexes[prim_index*3 + 0].uv * bary3.x + id.vertexes[prim_index*3 + 1].uv * bary3.y + id.vertexes[prim_index*3 + 2].uv * bary3.z;
                    
                    f_color = texture(nonuniformEXT(sampler2D(tex[t], s)), uv);
                }
            }
        ",
    }
}