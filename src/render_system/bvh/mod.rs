use vulkano::buffer::BufferContents;

pub mod build;
pub mod aabb;


#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct BvhNode {
    // the min bound
    pub min_or_v0: [f32; 3],
    // the max bound
    pub max_or_v1: [f32; 3],
    // v2
    pub v2: [f32; 3],
    // how much power is in this light node
    pub luminance: f32,
    // if this is 0xFFFFFFFF, then this is a leaf node
    pub left_node_idx: u32,
    // if left_node_idx is 0xFFFFFFFF, right_node_idx_or_prim_idx is an `Index`
    // if this BVH represents a bottom level BVH, then `Index` is a GLSL PrimitiveIndex
    // if this BVH represents a top level BVH, then `Index` is a GLSL InstanceID
    // otherwise, it is the index of the right node
    // if this is 0xFFFFFFFF and left_node_idx is 0xFFFFFFFF, then this means that this is a dummy node
    pub right_node_idx_or_prim_idx: u32,
}

impl BvhNode {
    pub fn dummy() -> BvhNode {
        BvhNode {
            min_or_v0: [0.0; 3],
            max_or_v1: [0.0; 3],
            v2: [0.0; 3],
            luminance: 0.0,
            left_node_idx: 0xFFFFFFFF,
            right_node_idx_or_prim_idx: 0xFFFFFFFF,
        }
    }
}