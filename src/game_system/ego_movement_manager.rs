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
        let UpdateData { ego_entity_id, .. } = data;
        let walk_impulse = if self.user_input_state.w {
            Vector3::new(1.0, 0.0, 0.0)
        } else if self.user_input_state.s {
            Vector3::new(-1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };
        let jump_impulse = if self.user_input_state.space {
            Vector3::new(0.0, 4.0, 0.0)
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };

        let velocity_impulse = walk_impulse + jump_impulse;

        let torque_impulse = if self.user_input_state.a {
            Vector3::new(0.0, -1.0, 0.0)
        } else if self.user_input_state.d {
            Vector3::new(0.0, 1.0, 0.0)
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };

        return vec![WorldChange::AddImpulseEntity {
            id: ego_entity_id,
            velocity: velocity_impulse,
            torque: torque_impulse,
        }];
    }

    fn handle_event(&mut self, _: [u32; 2], event: &winit::event::WindowEvent) {
        self.user_input_state.handle_input(event);
    }
}
