use vulkano::buffer::BufferContents;

pub mod aabb;
pub mod build;

#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct BvhNode {
    // if this is 0xFFFFFFFF, then this is a leaf node
    pub left_node_idx: u32,
    // if left_node_idx is 0xFFFFFFFF, right_node_idx_or_prim_idx is an `Index`
    // if this BVH represents a bottom level BVH, then `Index` is a GLSL PrimitiveIndex
    // if this BVH represents a top level BVH, then `Index` is a GLSL InstanceID
    // otherwise, it is the index of the right node
    // if this is 0xFFFFFFFF and left_node_idx is 0xFFFFFFFF, then this means that this is a dummy node
    pub right_node_idx_or_prim_idx: u32,
    // the min bound (if internal) or v0 (if BL leaf)
    pub min_or_v0: [f32; 3],
    // the max bound (if internal) or v1 (if BL leaf)
    pub max_or_v1: [f32; 3],
    // the luminance of the 6 faces of the AABB (if internal) or v2 (if BL leaf)
    pub left_luminance_or_v2_1: f32,
    pub right_luminance_or_v2_2: f32,
    pub down_luminance_or_v2_3: f32,
    // the luminance of the 6 faces of the AABB (if internal) or the luminance of the primitive (if BL leaf)
    pub up_luminance_or_prim_luminance: f32,
    // unused if BL leaf
    pub back_luminance: f32,
    pub front_luminance: f32,
}

impl Default for BvhNode {
    fn default() -> BvhNode {
        BvhNode {
            left_node_idx: 0xFFFFFFFF,
            right_node_idx_or_prim_idx: 0xFFFFFFFF,
            min_or_v0: [0.0; 3],
            max_or_v1: [0.0; 3],
            left_luminance_or_v2_1: 0.0,
            right_luminance_or_v2_2: 0.0,
            down_luminance_or_v2_3: 0.0,
            up_luminance_or_prim_luminance: 0.0,
            back_luminance: 0.0,
            front_luminance: 0.0,
        }
    }
}
