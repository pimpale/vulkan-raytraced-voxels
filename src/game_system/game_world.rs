use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use nalgebra::Isometry3;
use nalgebra::Point3;
use nalgebra::Vector3;

use rapier3d::geometry::Collider;
use threadpool::ThreadPool;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::DeviceOwned;
use vulkano::device::Queue;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::swapchain::Surface;

use crate::camera::InteractiveCamera;
use crate::game_system::block::BlockDefinitionTable;
use crate::game_system::block::BlockIdx;
use crate::game_system::chunk_manager::ChunkManager;
use crate::game_system::ego_controls_manager::EgoMovementManager;
use crate::game_system::manager::Manager;
use crate::game_system::manager::UpdateData;
use crate::game_system::physics_manager::PhysicsManager;
use crate::game_system::scene_manager::SceneManager;
use crate::render_system::interactive_rendering;
use crate::render_system::scene::Scene;
use crate::render_system::vertex::Vertex3D;

pub struct EntityCreationPhysicsData {
    // if true, the object can be moved by the physics engine
    // if false, then the object will not move due to forces.
    pub is_dynamic: bool,
    // If hitbox is specified, it can still be collided with even if is_dynamic is false
    pub hitbox: Collider,
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
    MoveEntity {
        id: u32,
        velocity: Vector3<f32>,
        torque: Vector3<f32>,
    },
    BreakBlock {
        origin: Point3<f32>,
        direction: Vector3<f32>,
    },
    AddBlock {
        origin: Point3<f32>,
        direction: Vector3<f32>,
        block_id: BlockIdx,
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
    events_since_last_step: Vec<winit::event::WindowEvent<'static>>,
    changes_since_last_step: Vec<WorldChange>,
    managers: Vec<Box<dyn Manager>>,
}

impl GameWorld {
    pub fn new(
        general_queue: Arc<Queue>,
        transfer_queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        ego_entity_id: u32,
        surface: Arc<Surface>,
        camera: Box<dyn InteractiveCamera>,
    ) -> GameWorld {
        let device = general_queue.device();

        assert!(device == memory_allocator.device());

        let mut texture_atlas = vec![];

        let block_table = Arc::new(BlockDefinitionTable::load_assets(
            "assets/blocks",
            &mut texture_atlas,
        ));

        let renderer = interactive_rendering::Renderer::new(
            surface.clone(),
            general_queue.clone(),
            command_buffer_allocator.clone(),
            memory_allocator.clone(),
            descriptor_set_allocator.clone(),
            texture_atlas,
        );

        let scene = Rc::new(RefCell::new(Scene::new(
            general_queue.clone(),
            transfer_queue.clone(),
            memory_allocator.clone(),
            command_buffer_allocator.clone(),
        )));

        let threadpool = Arc::new(ThreadPool::new(15));

        let scene_manager = SceneManager::new(scene.clone());

        let camera = Rc::new(RefCell::new(camera));

        let ego_movement_manager = EgoMovementManager::new(camera.clone());

        let physics_manager = PhysicsManager::new();

        let chunk_manager = ChunkManager::new(threadpool, 0, block_table);

        GameWorld {
            entities: HashMap::new(),
            scene,
            camera,
            ego_entity_id,
            renderer,
            surface,
            events_since_last_step: vec![],
            changes_since_last_step: vec![],
            managers: vec![
                Box::new(ego_movement_manager),
                Box::new(physics_manager),
                Box::new(chunk_manager),
                Box::new(scene_manager),
            ],
        }
    }

    fn get_reserve_closure<'a>(entities: &'a HashMap<u32, Entity>) -> impl FnMut() -> u32 + 'a {
        let reserved_ids = vec![];
        move || loop {
            let id = rand::random::<u32>();
            if !entities.contains_key(&id) && !reserved_ids.contains(&id) {
                return id;
            }
        }
    }

    pub fn update_entity_table(&mut self, changes: &Vec<WorldChange>) {
        for change in changes {
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
                    self.entities.remove(&entity_id);
                }
                WorldChange::UpdateEntityIsometry(entity_id, isometry) => {
                    if let Some(entity) = self.entities.get_mut(&entity_id) {
                        entity.isometry = isometry.clone();
                    }
                }
                _ => {}
            }
        }
    }

    pub fn step(&mut self) {
        let new_changes = {
            let extent = interactive_rendering::get_surface_extent(&self.surface);
            let mut reserve_fn = Self::get_reserve_closure(&self.entities);

            let mut new_changes = vec![];
            for manager in self.managers.iter_mut() {
                let data = UpdateData {
                    entities: &self.entities,
                    window_events: &self.events_since_last_step,
                    world_changes: &self.changes_since_last_step,
                    ego_entity_id: self.ego_entity_id,
                    extent,
                    reserve_entity_id: &mut reserve_fn,
                };
                // run each manager, and store the changes required
                new_changes.extend(manager.update(data));
            }
            new_changes
        };

        // clear window events
        self.events_since_last_step.clear();

        // update entity table
        self.update_entity_table(&new_changes);
        self.changes_since_last_step = new_changes;

        // render to screen
        let (eye, front, right, up) = self.camera.borrow().eye_front_right_up();
        let (
            top_level_acceleration_structure,
            instance_vertex_buffer_addresses,
            instance_transforms,
            build_future,
        ) = self.scene.borrow_mut().get_tlas();
        // render to screen
        self.renderer.render(
            build_future,
            top_level_acceleration_structure,
            instance_vertex_buffer_addresses,
            instance_transforms,
            eye,
            front,
            right,
            up,
        );

        // at this point we can now garbage collect the removed entities from the scene
        // this is because the renderer will block until the last frame has finished executing
        unsafe {
            self.scene.borrow_mut().dispose_old_objects();
        }
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

    pub fn handle_window_event(&mut self, input: winit::event::WindowEvent) {
        if let Some(event) = input.to_static() {
            self.events_since_last_step.push(event);
        }
    }
}
