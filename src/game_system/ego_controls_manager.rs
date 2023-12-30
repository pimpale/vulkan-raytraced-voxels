use std::{cell::RefCell, rc::Rc};

use nalgebra::Vector3;

use crate::{
    camera::InteractiveCamera, entity::WorldChange, handle_user_input::UserInputState, utils,
};

use super::manager::{Manager, UpdateData};

pub struct EgoMovementManager {
    camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
    user_input_state: UserInputState,
}

impl EgoMovementManager {
    pub fn new(camera: Rc<RefCell<Box<dyn InteractiveCamera>>>) -> Self {
        Self {
            camera,
            user_input_state: UserInputState::new(),
        }
    }
}

impl Manager for EgoMovementManager {
    fn update<'a>(&mut self, data: UpdateData<'a>, _: &Vec<WorldChange>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            extent,
            ..
        } = data;

        let ego = entities.get(&ego_entity_id).unwrap();

        // update camera
        let mut camera = self.camera.borrow_mut();
        camera.set_position(ego.isometry.translation.vector.into());
        camera.set_rotation(ego.isometry.rotation.into());
        let (cam_eye, cam_front, cam_right, cam_up) = camera.eye_front_right_up();

        let mut changes = vec![];

        // move

        let move_magnitude: f32 = 20.0;
        let rotate_magnitude: f32 = 2.0;
        let jump_magnitude: f32 = 20.0;

        let mut velocity = Vector3::new(0.0, 0.0, 0.0);
        let mut torque = Vector3::new(0.0, 0.0, 0.0);

        if self.user_input_state.w {
            velocity += move_magnitude * Vector3::new(1.0, 0.0, 0.0);
        }
        if self.user_input_state.s {
            velocity += move_magnitude * Vector3::new(-1.0, 0.0, 0.0);
        }

        if self.user_input_state.space {
            velocity += jump_magnitude * Vector3::new(0.0, 1.0, 0.0);
        };
        if self.user_input_state.shift {
            velocity += jump_magnitude * Vector3::new(0.0, -1.0, 0.0);
        };

        if self.user_input_state.a {
            torque += rotate_magnitude * Vector3::new(0.0, -1.0, 0.0);
        }
        if self.user_input_state.d {
            torque += rotate_magnitude * Vector3::new(0.0, 1.0, 0.0);
        }

        changes.push(WorldChange::MoveEntity {
            id: ego_entity_id,
            velocity: ego.isometry.rotation * velocity,
            torque,
        });

        // break blocks
        if self.user_input_state.mouse_right_down {            
            // find relative position on screen
            let mouse_pos = self.user_input_state.pos;
            let uv = utils::screen_to_uv(mouse_pos, extent);
            // get aspect ratio of scren
            let aspect = extent[1] as f32 / extent[0] as f32;

            // create a vector based on raycasting
            let direction = uv.x * cam_right * aspect + uv.y * cam_up + cam_front;

            changes.push(WorldChange::BreakBlock {
                origin: cam_eye,
                direction,
            });
        }

        changes
    }

    fn handle_event(&mut self, extent: [u32; 2], event: &winit::event::WindowEvent) {
        // update our state
        self.user_input_state.handle_input(event);

        // forward changes to camera too
        let mut camera = self.camera.borrow_mut();
        camera.handle_event(extent, event);
    }
}
