vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_ray_query: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_buffer_reference2: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_nonuniform_qualifier: require

#define M_PI 3.1415926535897932384626433832795
#define EPSILON_BLOCK 0.001

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler s;
layout(set = 0, binding = 1) uniform texture2D tex[];

layout(set = 1, binding = 0) uniform accelerationStructureEXT top_level_acceleration_structure;

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer Vertex {
    vec3 position;
    uint t;
    vec2 uv;
};

layout(buffer_reference, buffer_reference_align=4, scalar) readonly buffer BvhNode {
    uint left_node_idx;
    uint right_node_idx_or_prim_idx;
    vec3 min_or_v0;
    vec3 max_or_v1;
    float left_luminance_or_v2_1;
    float right_luminance_or_v2_2;
    float down_luminance_or_v2_3;
    float up_luminance_or_prim_luminance;
    float back_luminance;
    float front_luminance;
};

struct InstanceData {
    // points to the device address of the vertex data for this instance
    uint64_t vertex_buffer_addr;
    // points to the device address of the light bvh data for this instance
    uint64_t bvh_node_buffer_addr;
    // the transform of this instance
    mat4x3 transform;
};

layout(set = 1, binding = 1, scalar) readonly buffer InstanceDataBuffer {
    InstanceData instance_data[];
};

layout(set = 1, binding = 2, scalar) readonly buffer InputsRayOrigin {
    vec3 input_origin[];
};

layout(set = 1, binding = 3, scalar) readonly buffer InputsRayDirection {
    vec3 input_direction[];
};

layout(set = 1, binding = 4, scalar) writeonly buffer OutputsOrigin {
    vec3 output_origin[];
};

layout(set = 1, binding = 5, scalar) writeonly buffer OutputsDirection {
    vec3 output_direction[];
};

layout(set = 1, binding = 6, scalar) writeonly buffer OutputsEmissivity {
    vec3 output_emissivity[];
};

layout(set = 1, binding = 7, scalar) writeonly buffer OutputsReflectivity {
    vec3 output_reflectivity[];
};

layout(set = 1, binding = 8) writeonly buffer OutputsNeeMisWeight {
    float output_nee_mis_weight[];
};

layout(set = 1, binding = 9) writeonly buffer OutputsBsdfPdf {
    float output_bsdf_pdf[];
};

layout(set = 1, binding = 10) writeonly buffer OutputsDebugInfo {
    vec4 output_debug_info[];
};


layout(push_constant, scalar) uniform PushConstants {
    uint bounce_seed;
    uint xsize;
    uint ysize;
    uint64_t tl_bvh_addr;
};


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

// accepts a seed, h, and a 32 bit integer, k, and returns a 32 bit integer
// corresponds to the loop in the murmur3 hash algorithm
// the output should be passed to murmur3_finalize before being used
uint murmur3_combine(uint h, uint k) {
    // murmur3_32_scrambleBlBvhNodeBuffer
    k *= 0x1b873593;

    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 0xe6546b64;
    return h;
}

// accepts a seed, h and returns a random 32 bit integer
// corresponds to the last part of the murmur3 hash algorithm
uint murmur3_finalize(uint h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uint murmur3_combinef(uint h, float k) {
    return murmur3_combine(h, floatBitsToUint(k));
}

float murmur3_finalizef(uint h) {
    return floatConstruct(murmur3_finalize(h));
}

float lengthSquared(vec3 v) {
    return dot(v, v);
}

struct VisibleTriangles {
    uint num_visible;
    vec3[3] tri0;
    vec3[3] tri1;
};


vec3 line_plane_intersection(vec3 line_point, vec3 line_direction, vec3 plane_point, vec3 plane_normal) {
    float t = dot(plane_normal, line_point - plane_point) / dot(-line_direction, plane_normal);
    return line_point + t * line_direction;
}

// returns anywhere between 0 and 2 triangles, all of which are visible from the point in the direction of the normal
VisibleTriangles splitIntoVisibleTriangles(vec3 point, vec3 normal, vec3[3] tri) {
    vec3[3] tri_sorted = tri;

    // sort 3 vertices by cosine of angle between vertex and normal
    float cos0 = dot(tri_sorted[0]-point, normal);
    float cos1 = dot(tri_sorted[1]-point, normal);
    float cos2 = dot(tri_sorted[2]-point, normal);

    if(cos0 > cos2) {
        vec3 tmp = tri_sorted[0];
        tri_sorted[0] = tri_sorted[2];
        tri_sorted[2] = tmp;
        float tmp2 = cos0;
        cos0 = cos2;
        cos2 = tmp2;
    }

    if(cos0 > cos1) {
        vec3 tmp = tri_sorted[0];
        tri_sorted[0] = tri_sorted[1];
        tri_sorted[1] = tmp;
        float tmp2 = cos0;
        cos0 = cos1;
        cos1 = tmp2;
    }

    if(cos1 > cos2) {
        vec3 tmp = tri_sorted[1];
        tri_sorted[1] = tri_sorted[2];
        tri_sorted[2] = tmp;
        float tmp2 = cos1;
        cos1 = cos2;
        cos2 = tmp2;
    }

    vec3[3] null_tri = vec3[3](vec3(0.0), vec3(0.0), vec3(0.0));

    if(dot(tri_sorted[2]-point, normal) <= 0) {
        // none of the triangle's vertices are visible 
        return VisibleTriangles(0, null_tri, null_tri);
    } else if(dot(tri_sorted[1]-point, normal) <= 0) {
        // only one of the triangle's vertices (vertex 2) is visible
        // we can now construct a new triangle that is guaranteed to be visible by finding
        // the intersection of v2->v1 with the normal plane (new_v0)
        // and the intersection of v2->v0 with the normal plane (new_v1)

        // for the line plane intersection test, we would normally need to check if the determinant of the matrix
        // formed by the vectors is zero, but we know that the normal is not parallel to the plane, so we can skip that check

        vec3[3] tri0 = vec3[3](
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[1] - tri_sorted[2],
                point,
                normal
            ),
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[0] - tri_sorted[2],
                point,
                normal
            ),
            tri_sorted[2]
        );
        return VisibleTriangles(1, tri0, null_tri);
    } else if(dot(tri_sorted[0] - point, normal) <= 0) {
        // two of the triangle's vertices are visible
        
        // in this case we have two visible triangles:
        // the triangle formed by v2, v1, and the intersection of v2->v0 with the normal plane
        // and the triangle formed by v2, the intersection of v2->v0 with the normal plane, and the intersection of v1->v0 with the normal plane
        
        vec3[3] tri0 = vec3[3](
            tri_sorted[2],
            tri_sorted[1],
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[0] - tri_sorted[2],
                point,
                normal
            )
        );

        vec3[3] tri1 = vec3[3](
            tri_sorted[2],
            line_plane_intersection(
                tri_sorted[2],
                tri_sorted[0] - tri_sorted[2],
                point,
                normal
            ),
            line_plane_intersection(
                tri_sorted[1],
                tri_sorted[0] - tri_sorted[1],
                point,
                normal
            )
        );
        
        return VisibleTriangles(2, tri0, tri1);
    } else {
        // all of the triangle's vertices are visible
        // so return the original triangle
        return VisibleTriangles(1, tri, null_tri);
    }
}

// returns the area of the triangle that is visible from the point in the direction of the normal
float getVisibleTriangleArea(VisibleTriangles vt) {
    if(vt.num_visible == 1) {
        return 0.5*length(cross(vt.tri0[1] - vt.tri0[0], vt.tri0[2] - vt.tri0[0]));
    } else if(vt.num_visible == 2) {
        return 0.5*length(cross(vt.tri0[1] - vt.tri0[0], vt.tri0[2] - vt.tri0[0])) + 0.5*length(cross(vt.tri1[1] - vt.tri1[0], vt.tri1[2] - vt.tri1[0]));
    } else {
        return 0.0;
    }
}

// // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
// bool rayTriangleIntersect(
//     vec3 orig, vec3 dir,
//     vec3 v0, vec3 v1, vec3 v2,
//     out float t
// )
// {
//     const float EPS = 0.0000001;
//     const float EPS2 = 0.0001;


//     // Compute the plane's normal
//     vec3 v0v1 = v1 - v0;
//     vec3 v0v2 = v2 - v0;
//     // No need to normalize
//     vec3 N = cross(v0v1, v0v2); // N
//     float area2 = length(N);

//     // Step 1: Finding P
    
//     // Check if the ray and plane are parallel
//     float NdotRayDirection = dot(N, dir);
//     if (abs(NdotRayDirection) < EPS) // Almost 0
//         return false; // They are parallel, so they don't intersect!

//     // Compute d parameter using equation 2
//     float d = -dot(N, v0);
    
//     // Compute t (equation 3)
//     t = -(dot(N, orig) + d) / NdotRayDirection;
    
//     // Check if the triangle is behind the ray
//     if (t < 0) return false; // The triangle is behind

//     // Compute the intersection point using equation 1
//     vec3 P = orig + t * dir;

//     // Step 2: Inside-Outside Test
//     vec3 C; // Vector perpendicular to triangle's plane

//     // Edge 0
//     vec3 edge0 = v1 - v0; 
//     vec3 vp0 = P - v0;
//     C = cross(edge0, vp0);
//     if (dot(N, C) < -EPS2) return false; // P is on the right side

//     // Edge 1
//     vec3 edge1 = v2 - v1; 
//     vec3 vp1 = P - v1;
//     C = cross(edge1, vp1);
//     if (dot(N, C) < -EPS2) return false; // P is on the right side

//     // Edge 2
//     vec3 edge2 = v0 - v2; 
//     vec3 vp2 = P - v2;
//     C = cross(edge2, vp2);
//     if (dot(N, C) < -EPS2) return false; // P is on the right side

//     return true; // This ray hits the triangle
// }

// bool rayVisibleTriangleIntersect(
//     vec3 orig, vec3 dir,
//     VisibleTriangles vt,
//     out float t
// ) {
//     if(vt.num_visible == 0) {
//         return false;
//     } else {
//         bool success = rayTriangleIntersect(
//             orig, dir,
//             vt.tri0[0], vt.tri0[1], vt.tri0[2],
//             t
//         );
//         if(!success && vt.num_visible == 2) {
//             return rayTriangleIntersect(
//                 orig, dir,
//                 vt.tri1[0], vt.tri1[1], vt.tri1[2],
//                 t
//             );
//         } else {
//             return success;
//         }
//     }
// }

vec3[3] triangleTransform(mat4x3 transform, vec3[3] tri) {
    return vec3[3](
        transform * vec4(tri[0], 1.0),
        transform * vec4(tri[1], 1.0),
        transform * vec4(tri[2], 1.0)
    );
}

vec3 triangleCenter(vec3[3] tri) {
    return (tri[0] + tri[1] + tri[2]) / 3.0;
}

float triangleRadiusSquared(vec3 center, vec3[3] tri) {
    return max(
        max(
            lengthSquared(tri[0] - center),
            lengthSquared(tri[1] - center)
        ),
        lengthSquared(tri[2] - center)
    );
}

struct IntersectionCoordinateSystem {
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
};

IntersectionCoordinateSystem localCoordinateSystem(vec3[3] tri) {
    vec3 v0_1 = tri[1] - tri[0];
    vec3 v0_2 = tri[2] - tri[0];
    vec3 normal = cross(v0_1, v0_2);
    vec3 tangent = v0_1;
    vec3 bitangent = cross(normal, tangent);
    
    return IntersectionCoordinateSystem(
        normalize(normal),
        normalize(tangent),
        normalize(bitangent)
    );
}

// returns true if any part of the rect is visible from the point in the direction of the normal
bool rectIsVisible(vec3 point, vec3 normal, vec3[4] rect) {
    for(uint i = 0; i < 4; i++) {
        vec3 to_v = rect[i] - point;
        if(dot(to_v, normal) >= EPSILON_BLOCK) {
            return true;
        }
    }
    return false;
}

// gets the importance of a node relative to a point on a surface, specialized for leaf nodes
float nodeImportance(bool topLevel, vec3 point, vec3 normal, mat4x3 transform, BvhNode node) {
    // replace node with lower level node to get better bounds
    if(topLevel && node.left_node_idx == 0xFFFFFFFF) {
        InstanceData id = instance_data[node.right_node_idx_or_prim_idx];
        transform = id.transform;
        topLevel = false;
        node = BvhNode(id.bvh_node_buffer_addr);
    }
    
    if(topLevel || node.left_node_idx != 0xFFFFFFFF) {
        // get corners
        vec3 v000 = transform * vec4(node.min_or_v0, 1.0);
        vec3 v111 = transform * vec4(node.max_or_v1, 1.0);
        vec3 v001 = vec3(v000.x, v000.y, v111.z);
        vec3 v010 = vec3(v000.x, v111.y, v000.z);
        vec3 v011 = vec3(v000.x, v111.y, v111.z);
        vec3 v100 = vec3(v111.x, v000.y, v000.z);
        vec3 v101 = vec3(v111.x, v000.y, v111.z);
        vec3 v110 = vec3(v111.x, v111.y, v000.z);

        float distance_sq = max(
            lengthSquared(v111 - v000),
            lengthSquared(0.5*(v000 + v111) - point)
        );

        float luminance = 0.0;
        
        vec3 lv = v000 - v100;
        luminance +=
            // light contained in this part of the node
            node.left_luminance_or_v2_1
            // clamped scalar projection of point onto x axis (to reduce estimate if we are inside the node)
            * clamp(dot(point - v100, lv)/lengthSquared(lv), 0.0, 1.0)
            // very rough approximation of cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v100, v101, v111, v110)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v100), normalize(lv)));

        vec3 rv = v100 - v000;
        luminance +=
            node.right_luminance_or_v2_2 
            * clamp(dot(point - v000, rv)/lengthSquared(rv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v000, v001, v011, v010)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v000), normalize(rv)));
        
        vec3 dv = v000 - v010;
        luminance +=
            node.down_luminance_or_v2_3 
            * clamp(dot(point - v010, dv)/lengthSquared(dv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v010, v011, v111, v110)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v010), normalize(dv)));

        vec3 uv = v010 - v000;
        luminance +=
            node.up_luminance_or_prim_luminance
            * clamp(dot(point - v000, uv)/lengthSquared(uv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v000, v001, v101, v100)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v000), normalize(uv)));

        vec3 bv = v000 - v001;
        luminance +=
            node.back_luminance
            * clamp(dot(point - v001, bv)/lengthSquared(bv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v001, v011, v111, v101)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v001), normalize(bv)));


        vec3 fv = v001 - v000;
        luminance +=
            node.front_luminance
            * clamp(dot(point - v000, fv)/lengthSquared(fv), 0.0, 1.0)
            // cos theta1
            * float(rectIsVisible(point, normal, vec3[4](v000, v010, v110, v100)));
            // // cos theta2
            // * max(0.0, dot(normalize(point - v000), normalize(fv)));


        return luminance / distance_sq;
    } else {
        // untransformed triangle
        vec3[3] tri_r = vec3[3](
            node.min_or_v0,
            node.max_or_v1,
            vec3(
                node.left_luminance_or_v2_1,
                node.right_luminance_or_v2_2,
                node.down_luminance_or_v2_3
            )
        );
        // transformed triangle visible to us
        vec3[3] tri = triangleTransform(transform, tri_r);
        VisibleTriangles vt = splitIntoVisibleTriangles(point, normal, tri);
        if(vt.num_visible == 0) {
            return 0.0;
        }

        vec3 tri_centroid = vt.num_visible == 1 
            ? triangleCenter(vt.tri0)
            : 0.5*(triangleCenter(vt.tri0) + triangleCenter(vt.tri1));

        vec3 tri_normal = normalize(cross(tri[1] - tri[0], tri[2] - tri[0]));

        // get total luminance of the triangle
        float emitted_light = getVisibleTriangleArea(vt)*node.up_luminance_or_prim_luminance;
        
        // https://en.wikipedia.org/wiki/View_factor
        float dist_to_tri = length(point - tri_centroid);
        float cos_theta_tri = dot(tri_normal, point-tri_centroid)/dist_to_tri;
        float cost_theta_surf = dot(normal, tri_centroid-point)/dist_to_tri;
        if (cost_theta_surf < 0.0 || cos_theta_tri < 0.0) {
            return 0.0;
        }

        float min_distance_sq = triangleRadiusSquared(tri_centroid, tri);
        float distance_sq = max(dist_to_tri*dist_to_tri, min_distance_sq);
        
        float visibility_coefficient = cos_theta_tri*cost_theta_surf;

        return emitted_light*visibility_coefficient / distance_sq;
    }
}

struct BvhTraverseResult {
    bool success;
    uint instance_index;
    uint prim_index;
    float probability;
    float importance;
};

BvhTraverseResult traverseBvh(vec3 point, vec3 normal, uint seed) {
    BvhNode root = BvhNode(tl_bvh_addr);
    BvhNode node = root;

    // check that the top level bvh isn't a dummy node
    if(node.left_node_idx == 0xFFFFFFFF && node.right_node_idx_or_prim_idx == 0xFFFFFFFF) {
        return BvhTraverseResult(
            false,
            0,
            0,
            1.0,
            0.0
        );
    }

    float probability = 1.0;
    float importance = 0.0;
    mat4x3 transform = mat4x3(1.0);
    uint instance_index = 0xFFFFFFFF;
    bool topLevel = true;
    while(true) {
        if(topLevel && node.left_node_idx == 0xFFFFFFFF) {
            instance_index = node.right_node_idx_or_prim_idx;
            InstanceData id = instance_data[node.right_node_idx_or_prim_idx];
            transform = id.transform;
            topLevel = false;
            root = BvhNode(id.bvh_node_buffer_addr);
            node = root;
            if(importance == 0.0) {
                importance = nodeImportance(topLevel, point, normal, transform, node);
            }
        }
        if(!topLevel && node.left_node_idx == 0xFFFFFFFF) {
            return BvhTraverseResult(
                true,
                instance_index,
                node.right_node_idx_or_prim_idx,
                probability,
                importance
            );
        }

        // otherwise pick a child node
        BvhNode left = root[node.left_node_idx];
        BvhNode right = root[node.right_node_idx_or_prim_idx];

        float left_importance = nodeImportance(topLevel, point, normal, transform, left);
        float right_importance = nodeImportance(topLevel, point, normal, transform, right);
        float total_importance = left_importance + right_importance;
        float left_importance_normalized = left_importance / total_importance;
        float right_importance_normalized = right_importance / total_importance;

        if (total_importance == 0.0) {
            return BvhTraverseResult(
                false,
                0,
                0,
                probability,
                0.0
            );
        } else if(murmur3_finalizef(seed) < left_importance_normalized) {
            node = left;
            probability *= left_importance_normalized;
            importance = left_importance;
        } else {
            node = right;
            probability *= right_importance_normalized;
            importance = right_importance;
        }
        seed = murmur3_combine(seed, 0);
    }
}


vec3 debugPrim(uint instance_index, uint prim_index) {
    uint colseed = murmur3_combine(instance_index, prim_index);
    return normalize(vec3(
        murmur3_finalizef(murmur3_combine(colseed, 0)),
        murmur3_finalizef(murmur3_combine(colseed, 1)),
        murmur3_finalizef(murmur3_combine(colseed, 2))
    ));
}

vec3 debugBvh(vec3 point, vec3 normal, uint seed) {
    BvhTraverseResult result = traverseBvh(point, normal, seed);
    if(result.success) {
        return 0.1*result.importance*debugPrim(result.instance_index, result.prim_index);
    } else {
        return vec3(-1.0, -1.0, -1.0);
    }
}

// returns a vector sampled from the hemisphere with positive y
// sample is weighted by cosine of angle between sample and y axis
// https://cseweb.ucsd.edu/classes/sp17/cse168-a/CSE168_08_PathTracing.pdf
vec3 cosineWeightedSampleHemisphere(vec2 uv) {
    float z = uv.x;
    float r = sqrt(max(0, 1.0 - z));
    float phi = 2.0 * M_PI * uv.y;
    
    return vec3(r * cos(phi), sqrt(z), r * sin(phi));
}

// returns a point sampled from a triangle
// equal area sampling
vec3 triangleSample(vec2 uv, vec3[3] tri) {
    if(uv.x + uv.y > 1.0) {
        uv = vec2(1.0 - uv.x, 1.0 - uv.y);
    }
    vec3 bary = vec3(1.0 - uv.x - uv.y, uv.x, uv.y);
    return bary.x * tri[0] + bary.y * tri[1] + bary.z * tri[2];
}

// there must be at least 1 visible triangle
vec3 visibleTriangleSample(vec3 tuv, VisibleTriangles vt) {
    if(vt.num_visible == 1) {
        return triangleSample(tuv.xy, vt.tri0);
    } else {
        float area1 = 0.5*length(cross(vt.tri0[1] - vt.tri0[0], vt.tri0[2] - vt.tri0[0]));
        float area2 = 0.5*length(cross(vt.tri1[1] - vt.tri1[0], vt.tri1[2] - vt.tri1[0]));
        float total_area = area1 + area2;
        if(tuv.x < area1/total_area) {
            return triangleSample(tuv.yz, vt.tri0);
        } else {
            return triangleSample(tuv.yz, vt.tri1);
        }
    }
}

// returns a vector sampled from the hemisphere defined around the coordinate system defined by normal, tangent, and bitangent
// normal, tangent and bitangent form a right handed coordinate system 
vec3 alignedCosineWeightedSampleHemisphere(vec2 uv, IntersectionCoordinateSystem ics) {
    vec3 hemsam = cosineWeightedSampleHemisphere(uv);
    return normalize(hemsam.x * ics.tangent + hemsam.y * ics.normal + hemsam.z * ics.bitangent);
}

struct IntersectionInfo {
    bool miss;
    uint instance_index;
    uint prim_index;
    vec2 bary;
};

IntersectionInfo getIntersectionInfo(vec3 origin, vec3 direction) {
    const float t_min = EPSILON_BLOCK;
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
            true,
            0,
            0,
            vec2(0.0)
        );
    } else {
        return IntersectionInfo(
            false,
            rayQueryGetIntersectionInstanceIdEXT(ray_query, true),
            rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true),
            rayQueryGetIntersectionBarycentricsEXT(ray_query, true)
        );
    }
}

void main() {
    // return early if we are out of bounds
    if(gl_GlobalInvocationID.x >= xsize || gl_GlobalInvocationID.y >= ysize) {
        return;
    }
    
    // tensor layout: [sample, y, x, channel]
    const uint bid =  
            + gl_GlobalInvocationID.z   * ysize * xsize 
            + gl_GlobalInvocationID.y   * xsize 
            + gl_GlobalInvocationID.x; 
            
    const vec3 origin = input_origin[bid];
    const vec3 direction = input_direction[bid];
    const uint seed = murmur3_combine(bounce_seed, bid);

    // return early from terminal samples (ray direction is 0, 0, 0)
    if(length(direction) == 0.0) {
        output_origin[bid] = origin;
        output_direction[bid] = direction;
        output_emissivity[bid] = vec3(0.0);
        output_reflectivity[bid] = vec3(0.0);
        output_nee_mis_weight[bid] = 0.0;
        output_bsdf_pdf[bid] = 1.0;
        output_debug_info[bid] = vec4(0.0);
        return;
    }

    // get intersection info
    IntersectionInfo info = getIntersectionInfo(origin, direction);

    if(info.miss) {
        output_origin[bid] = origin;
        output_direction[bid] = vec3(0.0); // no direction (miss)
        output_emissivity[bid] = vec3(5.0); // sky color
        output_reflectivity[bid] = vec3(0.0);
        output_nee_mis_weight[bid] = 0.0;
        output_bsdf_pdf[bid] = 1.0;
        output_debug_info[bid] = vec4(0.0);
        return;
    }


    // get barycentric coordinates
    vec3 bary3 = vec3(1.0 - info.bary.x - info.bary.y,  info.bary.x, info.bary.y);

    // get the instance data for this instance
    InstanceData id = instance_data[info.instance_index];

    Vertex v0 = Vertex(id.vertex_buffer_addr)[info.prim_index*3 + 0];
    Vertex v1 = Vertex(id.vertex_buffer_addr)[info.prim_index*3 + 1];
    Vertex v2 = Vertex(id.vertex_buffer_addr)[info.prim_index*3 + 2];

    // triangle untransformed
    vec3[3] tri_r = vec3[3](
        v0.position,
        v1.position,
        v2.position
    );

    // transform triangle
    vec3[3] tri = triangleTransform(id.transform, tri_r);

    IntersectionCoordinateSystem ics = localCoordinateSystem(tri);

    // get the texture coordinates
    uint t = v0.t;
    vec2 uv = v0.uv * bary3.x + v1.uv * bary3.y + v2.uv * bary3.z;


    vec3 new_origin = tri[0] * bary3.x + tri[1] * bary3.y + tri[2] * bary3.z;
    vec3 new_direction;
    vec3 debuginfo;

    // fetch data
    vec4 tex0 = texture(nonuniformEXT(sampler2D(tex[t*3+0], s)), uv).rgba;
    vec4 tex1 = texture(nonuniformEXT(sampler2D(tex[t*3+1], s)), uv).rgba;
    vec4 tex2 = texture(nonuniformEXT(sampler2D(tex[t*3+2], s)), uv).rgba;

    // MIS weight for choosing the light
    float light_pdf_mis_weight = 0.0;

    // probability of choosing the ray given the BSDF
    float bsdf_pdf;

    vec3 reflectivity = tex0.rgb;
    float alpha = tex0.a;
    vec3 emissivity = 1000.0*tex1.rgb * -dot(direction, ics.normal);
    float metallicity = tex2.r;

    // decide whether to do specular (0), transmissive (1), or lambertian (2) scattering
    float scatter_kind_rand = murmur3_finalizef(murmur3_combine(seed, 0));
    if(scatter_kind_rand < metallicity) {
        // mirror scattering
        bsdf_pdf = 1.0;

        new_direction = reflect(
            direction,
            ics.normal
        );
    } else if (scatter_kind_rand < metallicity + (1.0-alpha)) {
        // transmissive scattering
        new_direction = direction;
        reflectivity = vec3(1.0);
        bsdf_pdf = 1.0;
    } else {
        // lambertian scattering
        reflectivity = reflectivity / M_PI;

        // try traversing the bvh
        BvhTraverseResult result = traverseBvh(new_origin, ics.normal, murmur3_combine(seed, 2));

        // we have a 0% chance of picking the light if our bvh traversal was unsuccessful
        // otherwise, the chance is proportional to the importance of our pick
        if(result.success && result.importance > 0.0) {
            // chance of picking the light if our bvh traversal was successful
            light_pdf_mis_weight = clamp(result.importance / 10.0, 0.0, 0.5);
            // light_pdf_mis_weight = 0.5;
        }

        // light sampling data that may or may not be valid
        VisibleTriangles vt; // visible triangles struct

        if(light_pdf_mis_weight > 0.0) {
            // get the instance data for this instance
            InstanceData id_light = instance_data[result.instance_index];

            Vertex v0_light = Vertex(id_light.vertex_buffer_addr)[result.prim_index*3 + 0];
            Vertex v1_light = Vertex(id_light.vertex_buffer_addr)[result.prim_index*3 + 1];
            Vertex v2_light = Vertex(id_light.vertex_buffer_addr)[result.prim_index*3 + 2];

            // triangle untransformed
            vec3[3] tri_light_r = vec3[3](
                v0_light.position,
                v1_light.position,
                v2_light.position
            );

            // transform triangle
            vec3[3] tri_light = triangleTransform(id_light.transform, tri_light_r);
            vt = splitIntoVisibleTriangles(new_origin, ics.normal, tri_light);
        }

        // randomly choose whether or not to sample the light
        float mis_rand = murmur3_finalizef(murmur3_combine(seed, 3));
        if(mis_rand < light_pdf_mis_weight) {
            // sample a point on the light
            vec3 tuv_light = vec3(
                murmur3_finalizef(murmur3_combine(seed, 4)),
                murmur3_finalizef(murmur3_combine(seed, 5)),
                murmur3_finalizef(murmur3_combine(seed, 6))
            );
            vec3 sampled_light_point = visibleTriangleSample(tuv_light, vt);

            new_direction = normalize(sampled_light_point - new_origin);
        } else {
            // cosine weighted hemisphere sample
            new_direction = alignedCosineWeightedSampleHemisphere(
                // random uv
                vec2(
                    murmur3_finalizef(murmur3_combine(seed, 4)),
                    murmur3_finalizef(murmur3_combine(seed, 5))
                ),
                // align it with the normal of the object we hit
                ics
            );
        }         

        // cosine of the angle made between the surface normal and the new direction
        float cos_theta = dot(new_direction, ics.normal);

        // what is the probability of picking this ray if we treated the surface as lambertian and randomly sampled from the BRDF?
        bsdf_pdf = cos_theta / M_PI;

        // // compute the ray pdf for the light
        // float ray_pdf_light = 0.0;
        // if(light_pdf_mis_weight > 0.0) {
        //     float t;
        //     if(rayVisibleTriangleIntersect(new_origin, new_direction, vt, t)) {
        //         vec3 sampled_light_point = new_origin + t*new_direction;
        //         float light_area = getVisibleTriangleArea(vt);
        //         float light_distance = length(sampled_light_point - new_origin);
        //         // what is the probability of picking this ray if we were picking a random point on the light?
        //         ray_pdf_light = light_distance*light_distance/(cos_theta*light_area);
        //     }
        // }
    }

    // compute data for this bounce
    output_origin[bid] = new_origin;
    output_direction[bid] = new_direction;
    output_emissivity[bid] = emissivity;
    output_reflectivity[bid] = reflectivity;
    output_nee_mis_weight[bid] = light_pdf_mis_weight;
    output_bsdf_pdf[bid] = bsdf_pdf;
    output_debug_info[bid] = vec4(0.0);
}
",
}
