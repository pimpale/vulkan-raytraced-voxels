
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

            #define M_PI 3.1415926535897932384626433832795

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

            layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer InstanceVertexBuffer {
                Vertex vertexes[];
            };

            layout(set = 1, binding = 1) readonly buffer InstanceVertexBufferAddresses {
                // one uint64 per instance that points to the device address of the data for that instance
                uint64_t instance_vertex_buffer_addrs[];
            };

            layout(set = 1, binding = 2, scalar) readonly buffer InstanceTransforms {
                mat4 instance_transforms[];
            };

            layout(push_constant, scalar) uniform Camera {
                vec3 eye;
                vec3 front;
                vec3 up;
                vec3 right;
                float aspect;
                uint frame;
            } camera;

            float random(vec2 uv, float seed) {
                return fract(sin(mod(dot(uv, vec2(12.9898, 78.233)) + 1113.1 * seed, M_PI)) * 43758.5453);
            }
              
            // returns a vector sampled from the hemisphere with positive y
            vec3 uniformSampleHemisphere(vec2 uv) {
                float z = uv.x;
                float r = sqrt(max(0, 1.0 - z * z));
                float phi = 2.0 * M_PI * uv.y;
              
                return vec3(r * cos(phi), z, r * sin(phi));
            }

            // returns a vector sampled from the hemisphere defined around the normal vector
            vec3 alignedUniformSampleHemisphere(vec2 uv, vec3 normal) {
                // define a right handed coordinate system around the normal
                vec3 tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.1)));
                vec3 bitangent = normalize(cross(normal, tangent));
              
                vec3 hemsam = uniformSampleHemisphere(uv);
              
                return normalize(hemsam.x * tangent + hemsam.y * normal + hemsam.z * bitangent);
            }

            struct IntersectionInfo {
                vec3 position;
                vec3 normal;
                vec2 uv;
                uint t;
                bool miss;
            };

            IntersectionInfo getIntersectionInfo(vec3 origin, vec3 direction) {
                const float t_min = 0.01;
                const float t_max = 1000.0;
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
                
                // if miss return miss
                if(rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                    return IntersectionInfo(
                        vec3(0.0),
                        vec3(0.0),
                        vec2(0.0),
                        0,
                        true
                    );
                }
                
                uint prim_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true);
                uint instance_index = rayQueryGetIntersectionInstanceIdEXT(ray_query, true);

                // get barycentric coordinates
                vec2 bary = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);
                vec3 bary3 = vec3(1.0 - bary.x - bary.y,  bary.x, bary.y);

                // get the instance data for this instance
                InstanceVertexBuffer id = InstanceVertexBuffer(instance_vertex_buffer_addrs[instance_index]);
                Vertex v0 = id.vertexes[prim_index*3 + 0];
                Vertex v1 = id.vertexes[prim_index*3 + 1];
                Vertex v2 = id.vertexes[prim_index*3 + 2];

                mat4 transform = instance_transforms[instance_index];

                // get the texture coordinates
                uint t = v0.t;
                vec2 uv = v0.uv * bary3.x + v1.uv * bary3.y + v2.uv * bary3.z;
                    
                // get normal 
                vec3 v0_1 = id.vertexes[prim_index*3 + 1].position - id.vertexes[prim_index*3 + 0].position;
                vec3 v0_2 = id.vertexes[prim_index*3 + 2].position - id.vertexes[prim_index*3 + 0].position;
                vec3 normal = normalize(cross(v0_1, v0_2));

                // get position
                vec3 position = v0.position * bary3.x + v1.position * bary3.y + v2.position * bary3.z;

                // transform normal and position
                position = (transform * vec4(position, 1.0)).xyz;
                normal = normalize((transform * vec4(normal, 0.0)).xyz);

                return IntersectionInfo(
                    position,
                    normal,
                    uv,
                    t,
                    false
                );
            }



            void main2() {
                vec3 origin = camera.eye;
                vec3 direction = normalize(in_uv.x * camera.right * camera.aspect + in_uv.y * camera.up + camera.front);
                IntersectionInfo info = getIntersectionInfo(origin, direction);
                
                if(info.miss) {
                    f_color = vec4(0.0, 0.0, 0.0, 1.0);
                } else {


                    // not a cosine weighted hemisphere sample
                    // new ray direction
                    vec3 new_direction = alignedUniformSampleHemisphere(
                        // random uv
                        vec2(random(in_uv, 0.0), random(in_uv+vec2(50.0, 10.0), 50.0)),
                        // align it with the normal of the object we hit
                        info.normal
                    );

                    f_color = vec4(texture(nonuniformEXT(sampler2D(tex[info.t], s)), info.uv).rgb, 1.0);
                    f_color += vec4(dot(new_direction, info.normal)-0.5, 0.0, 0.0, 0.0);
                }
            }

            const uint MAX_BOUNCES = 4;

            vec3 samplePixel(float seed) {
                // initial ray origin and direction
                vec3 origin = camera.eye;
                vec3 direction = normalize(in_uv.x * camera.right * camera.aspect + in_uv.y * camera.up + camera.front);

                vec3 bounce_emittance[MAX_BOUNCES];
                vec3 bounce_reflectance[MAX_BOUNCES];
                // the probability of bouncing in the direction that we chose
                float bounce_probability[MAX_BOUNCES];
                // the cosine of the angle between the normal and the direction we chose
                float bounce_cos_theta[MAX_BOUNCES];

                int max_bounce = -1;
                for(uint bounce = 0; bounce < MAX_BOUNCES; bounce++) {
                    IntersectionInfo info = getIntersectionInfo(origin, direction);

                    if(info.miss) {
                        bounce_emittance[bounce] = vec3(5.0);
                        bounce_reflectance[bounce] = vec3(0.0);
                        bounce_probability[bounce] = 1.0;
                        bounce_cos_theta[bounce] = 1.0;
                        max_bounce = int(bounce);
                        break;
                    }

                    // new ray origin
                    origin = info.position + info.normal * 0.001;

                    // not a cosine weighted hemisphere sample
                    // new ray direction
                    direction = alignedUniformSampleHemisphere(
                        // random uv
                        vec2(random(in_uv, 0.0), random(in_uv+vec2(50.0, 10.0), seed)),
                        // align it with the normal of the object we hit
                        info.normal
                    );

                    // compute data for this bounce
                    bounce_emittance[bounce] = vec3(0.0);
                    bounce_reflectance[bounce] = texture(nonuniformEXT(sampler2D(tex[info.t], s)), info.uv).rgb / M_PI;
                    bounce_probability[bounce] = 1.0 / (2.0 * M_PI);
                    bounce_cos_theta[bounce] = dot(direction, info.normal);
                }

                if(max_bounce == -1) {
                    max_bounce = int(MAX_BOUNCES-1);
                }

                // now assemble these samples into a final color
                vec3 color = vec3(0.0);
                for(int i = max_bounce; i >= 0; i--) {
                    color = bounce_emittance[i] + (color * bounce_reflectance[i] * bounce_cos_theta[i] / bounce_probability[i]); 
                }
                return color;
            }



            void main() {
                const uint SAMPLES_PER_PIXEL = 16;
                vec3 color = vec3(0.0);
                for (uint i = 0; i < SAMPLES_PER_PIXEL; i++) {
                    color += samplePixel(float(i));
                }
            
                f_color = vec4(color / float(SAMPLES_PER_PIXEL), 1.0);
            }
        ",
    }
}