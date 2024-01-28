use vulkano::buffer::BufferContents;

pub mod blas;
pub mod tlas;


#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct BlBvhNode {
    // the approximate center of the node
    centroid: [f32; 3],
    // the diagonal of the node
    diagonal: f32,
    // how much power is in this light node
    power: f32,
    // if this is 0xFFFFFFFF, then this is a leaf node
    left_node_idx: u32,
    // if left_node_idx is 0xFFFFFFFF, right_node_idx_or_prim_idx is a primitive index
    // otherwise, it is a right node index
    right_node_idx_or_prim_idx: u32,
}

#[derive(Debug, Clone, BufferContents)]
#[repr(C)]
pub struct TlBvhNode {
    // the approximate center of the node
    centroid: [f32; 3],
    // the diagonal of the node
    diagonal: f32,
    // how much light power is in this node
    power: f32,
    // if this is 0xFFFFFFFF, then this is a leaf node
    // otherwise, it is a left node index
    left_node_idx: u32,
    // if left_node_idx is 0xFFFFFFFF, then this is the index of the instance
    // otherwise, it is a right node index
    right_node_idx_or_instancce_idx: u32,
}