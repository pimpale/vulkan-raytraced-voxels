use vulkano::buffer::BufferContents;

pub mod build;
pub mod aabb;


#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct BvhNode {
    // the min bound
    pub min: [f32; 3],
    // the max bound
    pub max: [f32; 3],
    // how much power is in this light node
    pub luminance: f32,
    // if this is 0xFFFFFFFF, then this is a leaf node
    pub left_node_idx: u32,
    // if left_node_idx is 0xFFFFFFFF, right_node_idx_or_prim_idx is an `Index`
    // if this BVH represents a bottom level BVH, then `Index` is a GLSL PrimitiveIndex
    // if this BVH represents a top level BVH, then `Index` is a GLSL InstanceID
    // otherwise, it is the index of the right node
    pub right_node_idx_or_prim_idx: u32,
}