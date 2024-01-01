use nalgebra::Vector3;

pub struct MotionController {

}

impl MotionController {
    pub fn new() -> MotionController {
        MotionController {

        }
    }

    pub fn get_impulse(current_vel: Vector3<f32>, target_vel: Vector3<f32>, mass: f32) -> Vector3<f32> {
        let impulse = target_vel - current_vel;
        impulse * mass
    }
}