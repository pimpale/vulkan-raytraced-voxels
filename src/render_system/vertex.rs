use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Clone, Copy, Debug, BufferContents, Vertex, Default)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32_SFLOAT)]
    pub _dummy0: f32,
    // u and v are the texture coordinates
    #[format(R32G32B32_SFLOAT)]
    pub tuv: [f32; 3],
    #[format(R32_SFLOAT)]
    pub _dummy1: f32,
}

impl Vertex3D {
    pub fn new(position: [f32; 3], tuv: [f32; 3]) -> Vertex3D {
        Vertex3D {
            position,
            _dummy0: 0.0,
            tuv,
            _dummy1: 0.0,
        }
    }
}
