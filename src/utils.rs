use nalgebra::{Point3, Vector3};

use crate::render_system::vertex::Vertex3D as Vertex;

// get axis aligned bounding box
pub fn get_aabb(obj: &[Vertex]) -> Vector3<f32> {
    let mut min = Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX);
    let mut max = Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN);
    for v in obj.iter() {
        if v.position[0] < min[0] {
            min[0] = v.position[0];
        }
        if v.position[1] < min[1] {
            min[1] = v.position[1];
        }
        if v.position[2] < min[2] {
            min[2] = v.position[2];
        }
        if v.position[0] > max[0] {
            max[0] = v.position[0];
        }
        if v.position[1] > max[1] {
            max[1] = v.position[1];
        }
        if v.position[2] > max[2] {
            max[2] = v.position[2];
        }
    }
    max - min
}