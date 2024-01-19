use std::{cell::RefCell, rc::Rc, sync::Arc, time::Instant};

use nalgebra::{Point3, Vector3};
use rapier3d::{dynamics::RigidBodyType, math::AngularInertia};

use crate::{
    camera::{InteractiveCamera, RenderingPreferences},
    game_system::game_world::WorldChange,
    handle_user_input::UserInputState,
    utils,
};

use super::{
    block::{BlockDefinitionTable, BlockFace, BlockIdx},
    chunk_manager::ChunkQuerier,
    game_world::{EntityCreationData, EntityPhysicsData},
    manager::{Manager, UpdateData},
};

pub struct EgoControlsManager {
    camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
    chunk_querier: ChunkQuerier,
    block_definition_table: Arc<BlockDefinitionTable>,
    last_broke_block: Instant,
    last_placed_block: Instant,
    user_input_state: UserInputState,
    selected_block_id: BlockIdx,
}

impl EgoControlsManager {
    pub fn new(
        camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
        chunk_querier: ChunkQuerier,
        block_definition_table: Arc<BlockDefinitionTable>,
    ) -> Self {
        Self {
            camera,
            chunk_querier,
            block_definition_table,
            user_input_state: UserInputState::new(),
            last_broke_block: Instant::now(),
            last_placed_block: Instant::now(),
            selected_block_id: 3,
        }
    }

    fn update_selected_block_id(&mut self, events: &Vec<winit::event::WindowEvent>) {
        match UserInputState::last_key_pressed(
            events,
            &[
                winit::event::VirtualKeyCode::Key1,
                winit::event::VirtualKeyCode::Key2,
                winit::event::VirtualKeyCode::Key3,
                winit::event::VirtualKeyCode::Key4,
                winit::event::VirtualKeyCode::Key5,
                winit::event::VirtualKeyCode::Key6,
                winit::event::VirtualKeyCode::Key7,
                winit::event::VirtualKeyCode::Key8,
                winit::event::VirtualKeyCode::Key9,
            ],
        ) {
            Some(winit::event::VirtualKeyCode::Key1) => self.selected_block_id = 0,
            Some(winit::event::VirtualKeyCode::Key2) => self.selected_block_id = 1,
            Some(winit::event::VirtualKeyCode::Key3) => self.selected_block_id = 2,
            Some(winit::event::VirtualKeyCode::Key4) => self.selected_block_id = 3,
            Some(winit::event::VirtualKeyCode::Key5) => self.selected_block_id = 4,
            Some(winit::event::VirtualKeyCode::Key6) => self.selected_block_id = 5,
            Some(winit::event::VirtualKeyCode::Key7) => self.selected_block_id = 6,
            Some(winit::event::VirtualKeyCode::Key8) => self.selected_block_id = 7,
            Some(winit::event::VirtualKeyCode::Key9) => self.selected_block_id = 8,
            _ => {}
        }
    }
}

impl Manager for EgoControlsManager {
    fn update<'a>(&mut self, data: UpdateData<'a>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            extent,
            window_events,
            ..
        } = data;

        let ego = entities.get(&ego_entity_id).unwrap();
        let mut physics_data = ego.physics_data.clone().unwrap();

        // update user input state
        self.user_input_state.handle_input(window_events);
        self.update_selected_block_id(window_events);

        // update camera
        let mut camera = self.camera.borrow_mut();
        camera.set_root_position(ego.isometry.translation.vector.clone().cast().into());
        camera.set_root_rotation(ego.isometry.rotation.clone().cast().into());
        camera.handle_event(extent, window_events);
        if UserInputState::key_pressed(window_events, winit::event::VirtualKeyCode::R) {
            let current_prefs = camera.rendering_preferences();
            let new_samples = match current_prefs.samples {
                1 => 2,
                2 => 4,
                4 => 8,
                8 => 16,
                16 => 32,
                32 => 64,
                _ => 1,
            };
            camera.set_rendering_preferences(RenderingPreferences {
                samples: new_samples,
            });
        }

        let (cam_eye, cam_front, cam_right, cam_up) = camera.eye_front_right_up();

        let mut changes = vec![];

        // switch mode
        if UserInputState::key_pressed(window_events, winit::event::VirtualKeyCode::Tab) {
            let new_rigid_body_type = match physics_data.rigid_body_type {
                RigidBodyType::Dynamic => RigidBodyType::KinematicVelocityBased,
                _ => RigidBodyType::Dynamic,
            };
            physics_data.rigid_body_type = new_rigid_body_type;

            changes.push(WorldChange::GlobalEntityRemove(ego_entity_id));
            changes.push(WorldChange::GlobalEntityAdd(
                ego_entity_id,
                EntityCreationData {
                    physics: Some(physics_data.clone()),
                    mesh: ego.mesh.clone(),
                    isometry: ego.isometry.clone(),
                },
            ));
        }

        // move

        match physics_data.rigid_body_type {
            RigidBodyType::KinematicVelocityBased => {
                let move_magnitude: f32 = 10.0;
                let rotate_magnitude: f32 = 2.0;
                let jump_magnitude: f32 = 10.0;

                let mut target_linvel = Vector3::zeros();
                let mut target_angvel = Vector3::zeros();

                if self.user_input_state.current.w {
                    target_linvel += move_magnitude * Vector3::new(1.0, 0.0, 0.0);
                }
                if self.user_input_state.current.s {
                    target_linvel += move_magnitude * Vector3::new(-1.0, 0.0, 0.0);
                }

                if self.user_input_state.current.space {
                    target_linvel += jump_magnitude * Vector3::new(0.0, 1.0, 0.0);
                };
                if self.user_input_state.current.shift {
                    target_linvel += jump_magnitude * Vector3::new(0.0, -1.0, 0.0);
                };

                if self.user_input_state.current.a {
                    target_angvel += rotate_magnitude * Vector3::new(0.0, -1.0, 0.0);
                }
                if self.user_input_state.current.d {
                    target_angvel += rotate_magnitude * Vector3::new(0.0, 1.0, 0.0);
                }

                // if kinematic we can directly set the velocity
                changes.push(WorldChange::PhysicsSetVelocity {
                    id: ego_entity_id,
                    linvel: ego.isometry.rotation * target_linvel,
                    angvel: target_angvel,
                });
            }
            RigidBodyType::Dynamic => {
                let move_magnitude = 5.0;
                let rotate_magnitude = 2.0;
                let jump_magnitude = 7.0;

                let mut target_xvel = 0.0;
                let mut target_yvel = 0.0;
                let mut target_angvel = 0.0;

                if self.user_input_state.current.w {
                    target_xvel += move_magnitude;
                }
                if self.user_input_state.current.s {
                    target_xvel += -move_magnitude;
                }

                if self.user_input_state.current.space {
                    target_yvel += jump_magnitude;
                };
                if self.user_input_state.current.shift {
                    target_yvel += -jump_magnitude;
                };

                if self.user_input_state.current.a {
                    target_angvel += -rotate_magnitude;
                }
                if self.user_input_state.current.d {
                    target_angvel += rotate_magnitude;
                }

                // current linvel in local coordinates
                let current_linvel = ego.isometry.rotation.inverse() * physics_data.linvel;

                let mut impulse_necessary = (Vector3::new(target_xvel, target_yvel, 0.0)
                    - current_linvel)
                    * physics_data.hitbox.mass()
                    * 0.3;

                if target_yvel == 0.0 || !physics_data.grounded {
                    impulse_necessary[1] = 0.0;
                }

                let torque_impulse_necessary = (target_angvel - physics_data.angvel[1])
                    * physics_data.hitbox.mass_properties().principal_inertia()[1]
                    * 0.1;

                changes.push(WorldChange::PhysicsApplyImpulse {
                    id: ego_entity_id,
                    impulse: ego.isometry.rotation * impulse_necessary,
                    torque_impulse: Vector3::new(0.0, torque_impulse_necessary, 0.0),
                });
            }
            _ => {}
        }

        // highlighting and block manipulation

        let mouse_pos = self.user_input_state.current.pos;
        let uv = utils::screen_to_uv(mouse_pos, extent);
        // get aspect ratio of scren
        let aspect = extent[0] as f32 / extent[1] as f32;

        // create a vector based on raycasting
        let direction = (uv.x * cam_right * aspect + uv.y * cam_up + cam_front)
            .normalize()
            .cast();

        if let Some((global_coords, block_face)) =
            self.chunk_querier
                .trace_to_solid(&cam_eye.clone().cast(), &direction, 10.0)
        {
            if self.user_input_state.current.mouse_left_down
                && self.last_broke_block.elapsed().as_millis() > 300
            {
                let air = self.block_definition_table.block_idx("air").unwrap();

                changes.push(WorldChange::WorldSetBlock {
                    global_coords,
                    block_id: air,
                });

                self.last_broke_block = Instant::now();
            } else if self.user_input_state.current.mouse_right_down
                && self.last_placed_block.elapsed().as_millis() > 300
            {
                let block_coords = global_coords
                    + match block_face {
                        BlockFace::LEFT => Vector3::new(-1, 0, 0),
                        BlockFace::RIGHT => Vector3::new(1, 0, 0),
                        BlockFace::DOWN => Vector3::new(0, -1, 0),
                        BlockFace::UP => Vector3::new(0, 1, 0),
                        BlockFace::BACK => Vector3::new(0, 0, -1),
                        BlockFace::FRONT => Vector3::new(0, 0, 1),
                    };
                changes.push(WorldChange::WorldSetBlock {
                    global_coords: block_coords,
                    block_id: self.selected_block_id,
                });

                self.last_placed_block = Instant::now();
            }
        }

        changes
    }
}
