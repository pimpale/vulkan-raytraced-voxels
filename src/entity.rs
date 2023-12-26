use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use nalgebra::Isometry3;
use nalgebra::Vector3;

use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::DeviceOwned;
use vulkano::device::Queue;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;

use crate::camera::InteractiveCamera;
use crate::game_system::camera_manager::CameraManager;
use crate::game_system::ego_movement_manager::EgoMovementManager;
use crate::game_system::manager::Manager;
use crate::game_system::manager::UpdateData;
use crate::game_system::physics_manager::PhysicsManager;
use crate::game_system::scene_manager::SceneManager;
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

pub struct Entity {
    // mesh (untransformed)
    pub mesh: Vec<Vertex3D>,
    // transformation from origin
    pub isometry: Isometry3<f32>,
}

pub enum WorldChange {
    AddEntity(u32, EntityCreationData),
    RemoveEntity(u32),
    UpdateEntityIsometry(u32, Isometry3<f32>),
    AddImpulseEntity {
        id: u32,
        velocity: Vector3<f32>,
        torque: Vector3<f32>,
    },
}

pub struct GameWorld {
    entities: HashMap<u32, Entity>,
    ego_entity_id: u32,
    scene: Rc<RefCell<Scene<u32, Vertex3D>>>,
    camera: Rc<RefCell<Box<dyn InteractiveCamera>>>,
    surface: Arc<Surface>,
    renderer: interactive_rendering::Renderer,

    // manager data
    changes_since_last_step: Vec<WorldChange>,
    managers: Vec<Box<dyn Manager>>,
}

impl GameWorld {
    pub fn new(
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        ego_entity_id: u32,
        surface: Arc<Surface>,
        camera: Box<dyn InteractiveCamera>,
    ) -> GameWorld {
        let device = queue.device();

        assert!(device == memory_allocator.device());

        let renderer = interactive_rendering::Renderer::new(
            surface.clone(),
            queue.clone(),
            command_buffer_allocator.clone(),
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
        );

        let scene = Rc::new(RefCell::new(Scene::new(
            queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
        )));

        let scene_manager = SceneManager::new(scene.clone());

        let camera = Rc::new(RefCell::new(camera));

        let camera_manager = CameraManager::new(camera.clone());

        let ego_movement_manager = EgoMovementManager::new();

        let physics_manager = PhysicsManager::new();

        GameWorld {
            entities: HashMap::new(),
            scene,
            camera,
            ego_entity_id,
            renderer,
            surface,
            changes_since_last_step: vec![],
            managers: vec![
                Box::new(scene_manager),
                Box::new(camera_manager),
                Box::new(ego_movement_manager),
                Box::new(physics_manager),
            ],
        }
    }

    pub fn step(&mut self) {
        let mut new_changes = vec![];
        for manager in self.managers.iter_mut() {
            let data = UpdateData {
                entities: &self.entities,
                ego_entity_id: self.ego_entity_id,
            };
            // run each manager, and store the changes required
            new_changes.extend(manager.update(data, &self.changes_since_last_step));
        }

        // update entity table
        for change in &new_changes {
            match change {
                WorldChange::AddEntity(entity_id, entity_creation_data) => {
                    self.entities.insert(
                        *entity_id,
                        Entity {
                            mesh: entity_creation_data.mesh.clone(),
                            isometry: entity_creation_data.isometry.clone(),
                        },
                    );
                }
                WorldChange::RemoveEntity(entity_id) => {
                    self.entities.remove(entity_id);
                }
                WorldChange::UpdateEntityIsometry(entity_id, isometry) => {
                    if let Some(entity) = self.entities.get_mut(entity_id) {
                        entity.isometry = isometry.clone();
                    }
                }
                _ => {}
            }
        }

        self.changes_since_last_step = new_changes;
    }

    // add a new entity to the world
    pub fn add_entity(&mut self, entity_id: u32, entity_creation_data: EntityCreationData) {
        self.entities.insert(
            entity_id,
            Entity {
                mesh: entity_creation_data.mesh.clone(),
                isometry: entity_creation_data.isometry.clone(),
            },
        );
        self.changes_since_last_step
            .push(WorldChange::AddEntity(entity_id, entity_creation_data));
    }

    // remove an entity from the world
    pub fn remove_entity(&mut self, entity_id: u32) {
        self.entities.remove(&entity_id);
        self.changes_since_last_step
            .push(WorldChange::RemoveEntity(entity_id));
    }

    /// render to screen (if interactive rendering is enabled)
    /// Note that all offscreen rendering is done during `step`
    pub fn render(&mut self) {
        let (eye, front, right, up) = self.camera.borrow().eye_front_right_up();
        let (
            top_level_acceleration_structure,
            top_level_geometry_offset_buffer,
            top_level_vertex_buffer,
        ) = self.scene.borrow_mut().tlas();
        // render to screen
        self.renderer.render(
            top_level_acceleration_structure,
            top_level_geometry_offset_buffer,
            top_level_vertex_buffer,
            eye,
            front,
            right,
            up,
        )
    }

    pub fn handle_window_event(&mut self, input: &winit::event::WindowEvent) {
        let extent = interactive_rendering::get_surface_extent(&self.surface);
        for manager in self.managers.iter_mut() {
            manager.handle_event(extent, input);
        }
    }
}
