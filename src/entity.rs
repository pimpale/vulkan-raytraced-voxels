use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::Isometry3;
use nalgebra::Vector3;
use rapier3d::dynamics::CCDSolver;
use rapier3d::dynamics::ImpulseJointSet;
use rapier3d::dynamics::IntegrationParameters;
use rapier3d::dynamics::IslandManager;
use rapier3d::dynamics::MultibodyJointSet;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::dynamics::RigidBodySet;
use rapier3d::geometry::BroadPhase;
use rapier3d::geometry::ColliderBuilder;
use rapier3d::geometry::ColliderSet;
use rapier3d::geometry::NarrowPhase;
use rapier3d::pipeline::PhysicsPipeline;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::DeviceOwned;
use vulkano::device::Queue;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;

use crate::camera::InteractiveCamera;
use crate::handle_user_input::UserInputState;
use crate::object;
use crate::render_system::interactive_rendering;
use crate::render_system::scene::Scene;
use crate::render_system::vertex::Vertex3D;

pub struct EntityCreationPhysicsData {
    // if true, the object can be moved by the physics engine
    // if false, then the object will not move due to forces. If hitbox is specified, it can still be collided with
    pub is_dynamic: bool,
}

pub struct EntityCreationData {
    // if not specified then the object is visual only
    pub physics: Option<EntityCreationPhysicsData>,
    // mesh (untransformed)
    pub mesh: Vec<Vertex3D>,
    // initial transformation
    // position and rotation in space
    pub isometry: Isometry3<f32>,
}

struct Entity {
    // physics
    rigid_body_handle: Option<RigidBodyHandle>,
    // mesh (untransformed)
    mesh: Vec<Vertex3D>,
    // transformation from origin
    isometry: Isometry3<f32>,
}

struct PerWindowState {
    entity_id: u32,
    surface: Arc<Surface>,
    camera: Box<dyn InteractiveCamera>,
    renderer: interactive_rendering::Renderer,
}

pub struct GameWorld {
    entities: HashMap<u32, Entity>,
    scene: Scene<u32, Vertex3D>,
    // physics data
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    // state per window
    per_window_state: Option<PerWindowState>,
    // handle user input
    user_input_state: UserInputState,
}

pub struct InteractiveRenderingConfig {
    pub tracking_entity: u32,
    pub surface: Arc<Surface>,
    pub camera: Box<dyn InteractiveCamera>,
}

impl GameWorld {
    pub fn new(
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        interactive_rendering_config: Option<InteractiveRenderingConfig>,
    ) -> GameWorld {
        let device = queue.device();

        assert!(device == memory_allocator.device());

        // initialize interactive rendering if necessary
        let per_window_state = match interactive_rendering_config {
            Some(InteractiveRenderingConfig {
                tracking_entity,
                surface,
                camera,
            }) => {
                let renderer = interactive_rendering::Renderer::new(
                    surface.clone(),
                    queue.clone(),
                    command_buffer_allocator.clone(),
                    memory_allocator.clone(),
                    descriptor_set_allocator.clone(),
                );

                Some(PerWindowState {
                    entity_id: tracking_entity,
                    camera,
                    surface,
                    renderer,
                })
            }
            None => None,
        };

        // initialize scene
        let scene = Scene::new(
            queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
            HashMap::new(),
        );

        GameWorld {
            entities: HashMap::new(),
            scene,
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            per_window_state,
            user_input_state: UserInputState::new(),
        }
    }

    pub fn step(&mut self) {
        // step physics
        self.physics_pipeline.step(
            &Vector3::new(0.0, -9.81, 0.0),
            &IntegrationParameters::default(),
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );

        // update entity positions from physics and update mesh if necessary
        for (&entity_id, entity) in self.entities.iter_mut() {
            let new_isometry = match entity {
                Entity {
                    rigid_body_handle: Some(rigid_body_handle),
                    ..
                } => self.rigid_body_set[*rigid_body_handle].position(),
                Entity { ref isometry, .. } => isometry,
            };

            if new_isometry != &entity.isometry {
                entity.isometry = new_isometry.clone();
                self.scene
                    .update_object(entity_id, object::transform(&entity.mesh, &entity.isometry));
            }
        }

        // update the entity that the camera is tracking
        if let Some(ref mut per_window_state) = self.per_window_state {
            if let Some(Entity {
                rigid_body_handle: Some(handle),
                isometry,
                ..
            }) = self.entities.get(&per_window_state.entity_id)
            {
                let impulse = if self.user_input_state.w {
                    Vector3::new(1.0, 0.0, 0.0)
                } else if self.user_input_state.s {
                    Vector3::new(-1.0, 0.0, 0.0)
                } else {
                    Vector3::new(0.0, 0.0, 0.0)
                };
                let torque_impulse = if self.user_input_state.a {
                    Vector3::new(0.0, -1.0, 0.0)
                } else if self.user_input_state.d {
                    Vector3::new(0.0, 1.0, 0.0)
                } else {
                    Vector3::new(0.0, 0.0, 0.0)
                };
                self.rigid_body_set[*handle]
                    .apply_impulse((isometry.rotation * impulse) * 0.09, true);
                self.rigid_body_set[*handle].apply_torque_impulse(torque_impulse * 0.01, true)
            }
        }

        // update per-window interactive cameras (if necessary)
        if let Some(ref mut per_window_state) = self.per_window_state {
            if let Some(entity) = self.entities.get(&per_window_state.entity_id) {
                let isometry = entity.isometry;
                per_window_state
                    .camera
                    .set_position(isometry.translation.vector.into());
                per_window_state
                    .camera
                    .set_rotation(isometry.rotation.into());
                per_window_state.camera.update();
            }
        }
    }

    pub fn add_entity(&mut self, entity_id: u32, entity_creation_data: EntityCreationData) {
        let EntityCreationData {
            physics,
            mesh,
            isometry,
        } = entity_creation_data;

        // add to physics solver if necessary
        let rigid_body_handle = match physics {
            Some(EntityCreationPhysicsData { is_dynamic }) => {
                // cuboid constructor uses "half-extents", which is just half of the cuboid's width, height, and depth
                let hitbox = object::get_aabb(&mesh) / 2.0;
                let rigid_body = match is_dynamic {
                    true => RigidBodyBuilder::dynamic(),
                    false => RigidBodyBuilder::fixed(),
                }
                .position(isometry)
                .build();

                let collider = ColliderBuilder::cuboid(hitbox.x, hitbox.y, hitbox.z).build();

                let rigid_body_handle = self.rigid_body_set.insert(rigid_body);
                self.collider_set.insert_with_parent(
                    collider,
                    rigid_body_handle,
                    &mut self.rigid_body_set,
                );

                Some(rigid_body_handle)
            }
            None => None,
        };

        // add mesh to scene
        self.scene
            .add_object(entity_id, object::transform(&mesh, &isometry));

        self.entities.insert(
            entity_id,
            Entity {
                rigid_body_handle,
                mesh,
                isometry,
            },
        );
    }

    /// render to screen (if interactive rendering is enabled)
    /// Note that all offscreen rendering is done during `step`
    pub fn render(&mut self) {
        if let Some(ref mut per_window_state) = self.per_window_state {
            let extent = interactive_rendering::get_surface_extent(&per_window_state.surface);
            let (eye, front, right, up) = per_window_state.camera.eye_front_right_up();
            per_window_state.renderer.render(
                self.scene.top_level_acceleration_structure(),
                eye,
                front,
                right,
                up,
            )
        }
    }

    pub fn remove_entity(&mut self, entity_id: u32) {
        let entity = self.entities.remove(&entity_id);
        match entity {
            Some(Entity {
                rigid_body_handle: Some(rigid_body_handle),
                ..
            }) => {
                self.rigid_body_set.remove(
                    rigid_body_handle,
                    &mut self.island_manager,
                    &mut self.collider_set,
                    &mut self.impulse_joint_set,
                    &mut self.multibody_joint_set,
                    true,
                );
            }
            _ => (),
        }
        self.scene.remove_object(entity_id);
    }

    pub fn handle_window_event(&mut self, input: &winit::event::WindowEvent) {
        self.user_input_state.handle_input(input);
        match self.per_window_state {
            Some(ref mut per_window_state) => {
                per_window_state.camera.handle_event(
                    interactive_rendering::get_surface_extent(&per_window_state.surface),
                    input,
                );
            }
            None => (),
        }
    }
}
