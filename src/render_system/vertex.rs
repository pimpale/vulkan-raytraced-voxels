use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Clone, Copy, Debug, BufferContents, Vertex, Default)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32_UINT)]
    pub t: u32,
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], tuv: [f32; 3]) -> Vertex3D {
        Vertex3D {
            position,
            t: tuv[2] as u32,
            uv: [tuv[0], tuv[1]],
        }
    }

    pub fn new2(position: [f32; 3], t: u32, uv: [f32; 2]) -> Vertex3D {
        Vertex3D {
            position,
            t,
            uv,
        }
    }
}

#[derive(Clone, Copy, Debug, BufferContents)]
#[repr(C)]
pub struct InstanceData {
    pub vertex_buffer_addr: u64,
    pub bvh_node_buffer_addr: u64,
    pub transform: [[f32; 3]; 4],
}