use std::{cell::RefCell, rc::Rc, time::Instant};

use nalgebra::Vector3;

use crate::{
    camera::InteractiveCamera,game_system::game_world::WorldChange, handle_user_input::UserInputState, utils,
};

use super::{manager::{Manager, UpdateData}, block::BlockIdx};

pub struct EgoMovementManager {
    camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
    last_broke_block: Instant,
    last_placed_block: Instant,
    user_input_state: UserInputState,
    selected_block_id: BlockIdx,
}

impl EgoMovementManager {
    pub fn new(camera: Rc<RefCell<Box<dyn InteractiveCamera>>>) -> Self {
        Self {
            user_input_state: UserInputState::new(),
            last_broke_block: Instant::now(),
            last_placed_block: Instant::now(),
            selected_block_id: 3,
            camera,
        }
    }
}

impl Manager for EgoMovementManager {
    fn update<'a>(&mut self, data: UpdateData<'a>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            extent,
            window_events,
            ..
        } = data;

        let ego = entities.get(&ego_entity_id).unwrap();

        // update camera
        let mut camera = self.camera.borrow_mut();
        camera.set_root_position(ego.isometry.translation.vector.into());
        camera.set_root_rotation(ego.isometry.rotation.into());
        camera.handle_event(extent, window_events);

        let (cam_eye, cam_front, cam_right, cam_up) = camera.eye_front_right_up();

        // update user input state
        self.user_input_state.handle_input(window_events);

        let mut changes = vec![];

        // move

        let move_magnitude: f32 = 20.0;
        let rotate_magnitude: f32 = 2.0;
        let jump_magnitude: f32 = 20.0;

        let mut velocity = Vector3::new(0.0, 0.0, 0.0);
        let mut torque = Vector3::new(0.0, 0.0, 0.0);

        if self.user_input_state.current.w {
            velocity += move_magnitude * Vector3::new(1.0, 0.0, 0.0);
        }
        if self.user_input_state.current.s {
            velocity += move_magnitude * Vector3::new(-1.0, 0.0, 0.0);
        }

        if self.user_input_state.current.space {
            velocity += jump_magnitude * Vector3::new(0.0, 1.0, 0.0);
        };
        if self.user_input_state.current.shift {
            velocity += jump_magnitude * Vector3::new(0.0, -1.0, 0.0);
        };

        if self.user_input_state.current.a {
            torque += rotate_magnitude * Vector3::new(0.0, -1.0, 0.0);
        }
        if self.user_input_state.current.d {
            torque += rotate_magnitude * Vector3::new(0.0, 1.0, 0.0);
        }

        changes.push(WorldChange::MoveEntity {
            id: ego_entity_id,
            velocity: ego.isometry.rotation * velocity,
            torque,
        });

        // break blocks
        if self.user_input_state.current.mouse_left_down
            && self.last_broke_block.elapsed().as_millis() > 300
        {
            // find relative position on screen
            let mouse_pos = self.user_input_state.current.pos;
            let uv = utils::screen_to_uv(mouse_pos, extent);
            // get aspect ratio of scren
            let aspect = extent[1] as f32 / extent[0] as f32;

            // create a vector based on raycasting
            let direction = uv.x * cam_right * aspect + uv.y * cam_up + cam_front;

            changes.push(WorldChange::BreakBlock {
                origin: cam_eye,
                direction,
            });

            self.last_broke_block = Instant::now();
        } else if self.user_input_state.current.mouse_right_down
            && self.last_placed_block.elapsed().as_millis() > 300
        {
            // find relative position on screen
            let mouse_pos = self.user_input_state.current.pos;
            let uv = utils::screen_to_uv(mouse_pos, extent);
            // get aspect ratio of scren
            let aspect = extent[1] as f32 / extent[0] as f32;

            // create a vector based on raycasting
            let direction = uv.x * cam_right * aspect + uv.y * cam_up + cam_front;

            changes.push(WorldChange::AddBlock {
                origin: cam_eye,
                direction,
                block_id: self.selected_block_id,
            });
        }

        changes
    }
}
