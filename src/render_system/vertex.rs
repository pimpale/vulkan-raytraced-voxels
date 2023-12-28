use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(Clone, Copy, Debug, BufferContents, Vertex, Default)]
#[repr(C)]
pub struct Vertex3D {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    // u and v are the texture coordinates
    #[format(R32G32B32_SFLOAT)]
    pub tuv: [f32; 3],
}

impl Vertex3D {
    pub fn new(position: [f32; 3], tuv: [f32; 3]) -> Vertex3D {
        Vertex3D {
            position,
            tuv,
        }
    }

    pub fn new2(position: [f32; 3], t: u32, uv: [f32; 2]) -> Vertex3D {
        let tuv = [t as f32, uv[0], uv[1]];
        Vertex3D::new(position, tuv)
    }
}
