use std::ops::AddAssign;

use nalgebra::Vector3;

use crate::{entity::WorldChange, handle_user_input::UserInputState};

use super::manager::{Manager, UpdateData};

pub struct EgoMovementManager {
    user_input_state: UserInputState,
}

impl EgoMovementManager {
    pub fn new() -> Self {
        Self {
            user_input_state: UserInputState::new(),
        }
    }
}

impl Manager for EgoMovementManager {
    fn update<'a>(&mut self, data: UpdateData<'a>, _: &Vec<WorldChange>) -> Vec<WorldChange> {
        let UpdateData { ego_entity_id, entities, .. } = data;

        let ego_isometry = entities.get(&ego_entity_id).unwrap().isometry;

        let move_magnitude: f32 = 0.09;
        let rotate_magnitude: f32 = 0.01;
        let jump_magnitude: f32 = 0.3;

        let mut velocity = Vector3::new(0.0, 0.0, 0.0);
        let mut torque = Vector3::new(0.0, 0.0, 0.0);

        if self.user_input_state.w {
            velocity += move_magnitude*Vector3::new(1.0, 0.0, 0.0);
        }
        if self.user_input_state.s {
            velocity += move_magnitude*Vector3::new(-1.0, 0.0, 0.0);
        }

        if self.user_input_state.space {
            velocity += jump_magnitude*Vector3::new(0.0, 1.0, 0.0);
        };

        if self.user_input_state.a {
            torque += rotate_magnitude*Vector3::new(0.0, -1.0, 0.0);
        }
        if self.user_input_state.d {
            torque += rotate_magnitude*Vector3::new(0.0, 1.0, 0.0);
        }

        

        if torque.norm() > 0.0 || velocity.norm() > 0.0 {
            return vec![WorldChange::AddImpulseEntity {
                id: ego_entity_id,
                velocity: ego_isometry.rotation*velocity,
                torque,
            }];
        } else {
            return vec![];
        }
    }

    fn handle_event(&mut self, _: [u32; 2], event: &winit::event::WindowEvent) {
        self.user_input_state.handle_input(event);
    }
}
