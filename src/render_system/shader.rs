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

            // A single iteration of Bob Jenkins' One-At-A-Time hashing algorithm.
            uint hash( uint x ) {
                x += ( x << 10u );
                x ^= ( x >>  6u );
                x += ( x <<  3u );
                x ^= ( x >> 11u );
                x += ( x << 15u );
                return x;
            }

            uint hashFloat(float x) {
                return hash(floatBitsToUint(x));
            }

            // source: https://stackoverflow.com/questions/4200224/random-noise-functions-for-glsl
            // Construct a float with half-open range [0:1] using low 23 bits.
            // All zeroes yields 0.0, all ones yields the next smallest representable value below 1.0.
            float floatConstruct( uint m ) {
                const uint ieeeMantissa = 0x007FFFFFu; // binary32 mantissa bitmask
                const uint ieeeOne      = 0x3F800000u; // 1.0 in IEEE binary32
            
                m &= ieeeMantissa;                     // Keep only mantissa bits (fractional part)
                m |= ieeeOne;                          // Add fractional part to 1.0
            
                float  f = uintBitsToFloat( m );       // Range [1:2]
                return f - 1.0;                        // Range [0:1]
            }

            vec2 randVec2(uint seed) {
                return vec2(floatConstruct(seed), floatConstruct(hash(seed)));
            }
              
            // returns a vector sampled from the hemisphere with positive y
            // https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_08_PathTracing.pdf
            vec3 cosineWeightedSampleHemisphere(vec2 uv) {
                float z = uv.x;
                float r = sqrt(max(0, 1.0 - z));
                float phi = 2.0 * M_PI * uv.y;
              
                return vec3(r * cos(phi), sqrt(z), r * sin(phi));
            }

            struct IntersectionCoordinateSystem {
                vec3 normal;
                vec3 tangent;
                vec3 bitangent;
            };


            // returns a vector sampled from the hemisphere defined around the coordinate system defined by normal, tangent, and bitangent
            // normal, tangent and bitangent form a right handed coordinate system 
            vec3 alignedCosineWeightedSampleHemisphere(vec2 uv, IntersectionCoordinateSystem ics) {
                vec3 hemsam = cosineWeightedSampleHemisphere(uv);
                return normalize(hemsam.x * ics.tangent + hemsam.y * ics.normal + hemsam.z * ics.bitangent);
            }


            struct IntersectionInfo {
                IntersectionCoordinateSystem hit_coords;
                vec3 position;
                vec2 uv;
                uint t;
                bool miss;
            };

            IntersectionInfo getIntersectionInfo(vec3 origin, vec3 direction) {
                const float t_min = 0.001;
                const float t_max = 1000.0;
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
                
                // if miss return miss
                if(rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
                    return IntersectionInfo(
                        IntersectionCoordinateSystem(
                            vec3(0.0),
                            vec3(0.0),
                            vec3(0.0)
                        ),
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

                // get the transformed positions
                vec3 v0_p = (transform * vec4(v0.position, 1.0)).xyz;
                vec3 v1_p = (transform * vec4(v1.position, 1.0)).xyz;
                vec3 v2_p = (transform * vec4(v2.position, 1.0)).xyz;

                // get the texture coordinates
                uint t = v0.t;
                vec2 uv = v0.uv * bary3.x + v1.uv * bary3.y + v2.uv * bary3.z;
                    
                // get normal 
                vec3 v0_1 = v1_p - v0_p;
                vec3 v0_2 = v2_p - v0_p;
                vec3 normal = cross(v0_1, v0_2);
                vec3 tangent = v0_1;
                vec3 bitangent = cross(normal, tangent);

                // get position
                vec3 position = v0_p * bary3.x + v1_p * bary3.y + v2_p * bary3.z;

                return IntersectionInfo(
                    IntersectionCoordinateSystem(
                        normalize(normal),
                        normalize(tangent),
                        normalize(bitangent)
                    ),
                    position,
                    uv,
                    t,
                    false
                );
            }

            struct BounceInfo {
                vec3 emittance;
                vec3 reflectance;
                bool miss;
                vec3 new_origin;
                vec3 new_direction;
                float scatter_pdf_over_ray_pdf;
            };

            BounceInfo doBounce(vec3 origin, vec3 direction, IntersectionInfo info, uint seed) {
                if(info.miss) {
                    vec3 sky_emittance = vec3(20.0);
                    vec3 sky_reflectance = vec3(0.0);
                    return BounceInfo(
                        sky_emittance,
                        sky_reflectance,
                        // miss, so the ray is done
                        true,
                        vec3(0.0),
                        vec3(0.0),
                        1.0
                    );
                }

                vec3 new_origin = info.position;

                vec3 new_direction;

                float scatter_pdf_over_ray_pdf;
                vec3 emittance;
                vec3 reflectance; 

                if(info.t == 2) {
                    // mirror scattering
                    emittance = vec3(0.0);
                    reflectance = vec3(1.0);
                    scatter_pdf_over_ray_pdf = 1.0;

                    new_direction = reflect(
                        direction,
                        info.hit_coords.normal
                    );

                } else {
                    // lambertian scattering

                    // cosine weighted hemisphere sample
                    new_direction = alignedCosineWeightedSampleHemisphere(
                        // random uv
                        randVec2(seed),
                        // align it with the normal of the object we hit
                        info.hit_coords
                    );


                    // for lambertian surfaces, the scatter pdf and the ray sampling pdf are the same
                    // see here: https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html#lightscattering/thescatteringpdf
                    scatter_pdf_over_ray_pdf = 1.0;

                    reflectance = texture(nonuniformEXT(sampler2D(tex[info.t], s)), info.uv).rgb / M_PI;
                    emittance = vec3(0.0);
                    
                    // for testing
                    if(reflectance.r > 0.2) {
                        emittance = vec3(5.0);
                    }
                }

                // compute data for this bounce
                return BounceInfo(
                    emittance,
                    reflectance,
                    false,
                    new_origin,
                    new_direction,
                    scatter_pdf_over_ray_pdf
                );
            }

            const uint SAMPLES_PER_PIXEL = 64;
            const uint MAX_BOUNCES = 4;

            void main() {
                uint pixel_seed = hash(camera.frame) ^ hashFloat(in_uv.x) ^ hashFloat(in_uv.y);
    
                vec3 bounce_emittance[MAX_BOUNCES];
                vec3 bounce_reflectance[MAX_BOUNCES];
                float bounce_scatter_pdf_over_ray_pdf[MAX_BOUNCES];

                // initial ray origin and direction
                vec3 first_origin = camera.eye;
                vec3 first_direction = normalize(in_uv.x * camera.right * camera.aspect + in_uv.y * camera.up + camera.front);
                
                // do the first cast, which is deterministic
                IntersectionInfo first_intersection_info = getIntersectionInfo(first_origin, first_direction);

                vec3 color = vec3(0.0);
                for (uint sample_id = 0; sample_id < SAMPLES_PER_PIXEL; sample_id++) {
                    uint sample_seed = pixel_seed ^ hash(sample_id);
                    // store first bounce data
                    BounceInfo bounce_info = doBounce(first_origin, first_direction, first_intersection_info, sample_seed);
                    bounce_emittance[0] = bounce_info.emittance;
                    bounce_reflectance[0] = bounce_info.reflectance;
                    bounce_scatter_pdf_over_ray_pdf[0] = bounce_info.scatter_pdf_over_ray_pdf;

                    vec3 origin = bounce_info.new_origin;
                    vec3 direction = bounce_info.new_direction;

                    uint current_bounce;
                    for (current_bounce = 1; current_bounce < MAX_BOUNCES; current_bounce++) {
                        IntersectionInfo intersection_info = getIntersectionInfo(origin, direction);
                        bounce_info = doBounce(origin, direction, intersection_info, sample_seed ^ hash(current_bounce));
                        bounce_emittance[current_bounce] = bounce_info.emittance;
                        bounce_reflectance[current_bounce] = bounce_info.reflectance;
                        bounce_scatter_pdf_over_ray_pdf[current_bounce] = bounce_info.scatter_pdf_over_ray_pdf;

                        if(bounce_info.miss) {
                            current_bounce++;
                            break;
                        }

                        origin = bounce_info.new_origin;
                        direction = bounce_info.new_direction;
                    }
                    
                    // compute the color for this sample
                    vec3 sample_color = vec3(0.0);
                    for(int i = int(current_bounce)-1; i >= 0; i--) {
                        sample_color = bounce_emittance[i] + (sample_color * bounce_reflectance[i] * bounce_scatter_pdf_over_ray_pdf[i]); 
                    }
                    color += sample_color;
                }
            
                // average the samples
                f_color = vec4(color / float(SAMPLES_PER_PIXEL), 1.0);
            }
        ",
    }
}
