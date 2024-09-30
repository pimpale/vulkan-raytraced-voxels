vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    src: r"
#version 460
#extension GL_EXT_scalar_block_layout: require

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, scalar) writeonly buffer OutputsOrigin {
    vec3 output_origin[];
};

layout(set = 0, binding = 1, scalar) writeonly buffer OutputsDirection {
    vec3 output_direction[];
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
    uint frame_seed;
} push_constants;


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

vec2 screen_to_uv(uvec2 screen, uvec2 screen_size) {
    return 2*vec2(screen)/vec2(screen_size) - 1.0;
}

void main() {
    Camera camera = push_constants.camera;
    if(gl_GlobalInvocationID.x >= camera.screen_size.x || gl_GlobalInvocationID.y >= camera.screen_size.y) {
        return;
    }

    const uint xsize = camera.screen_size.x;
    const uint ysize = camera.screen_size.y;

    // tensor layout: [bounce, sample, y, x, channel]
    const uint bid =
            + gl_GlobalInvocationID.z   * ysize * xsize 
            + gl_GlobalInvocationID.y   * xsize 
            + gl_GlobalInvocationID.x; 

    uint seed = murmur3_combine(push_constants.frame_seed, bid);

    // initial ray origin and direction
    vec2 uv = screen_to_uv(gl_GlobalInvocationID.xy, camera.screen_size);
    float aspect = float(camera.screen_size.x) / float(camera.screen_size.y);

    vec2 jitter = 0.01*vec2(
        (1.0/camera.screen_size.x)*(murmur3_finalizef(murmur3_combine(seed, 0))-0.5),
        (1.0/camera.screen_size.y)*(murmur3_finalizef(murmur3_combine(seed, 1))-0.5)
    );

    output_origin[bid] = camera.eye;
    output_direction[bid] = normalize((uv.x + jitter.x) * camera.right * aspect + (uv.y + jitter.y) * camera.up + camera.front);
}
",
}