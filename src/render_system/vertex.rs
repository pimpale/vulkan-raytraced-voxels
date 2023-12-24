use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], color: [f32; 4]) -> Vertex3D {
        Vertex3D { position, color }
    }
}