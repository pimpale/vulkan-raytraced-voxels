use nalgebra::Vector3;
use rapier3d::math::AngularInertia;

pub struct MotionController {
    mass: f32,
    angular_inertia: AngularInertia<f32>,
}

impl MotionController {
    pub fn new(mass: f32, angular_inertia: AngularInertia<f32>) -> MotionController {
        MotionController { mass, angular_inertia }
    }

    pub fn set_mass(&mut self, mass:f32) {
        self.mass = mass;
    }

    pub fn update(
        &self,
        current_linvel: Vector3<f32>,
        current_angvel: Vector3<f32>,
        target_linvel: Vector3<f32>,
        target_angvel: Vector3<f32>,
    ) -> (Vector3<f32>, Vector3<f32>) {
        let impulse = (target_linvel - current_linvel)*self.mass;
        let torque_impulse = (target_angvel - current_angvel);
        (impulse, torque_impulse)
    }
}
