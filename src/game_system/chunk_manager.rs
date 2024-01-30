use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::Instant,
};

use nalgebra::{Isometry3, Point3, Vector3};
use noise::{NoiseFn, OpenSimplex};
use rapier3d::{dynamics::RigidBodyType, geometry::Collider};
use threadpool::ThreadPool;
use vulkano::memory::allocator::MemoryAllocator;

use crate::{
    game_system::game_world::{EntityCreationData, EntityPhysicsData, WorldChange},
    render_system::{scene::{self, SceneUploadedObjectHandle, SceneUploader}, vertex::Vertex3D},
};

use super::{
    block::{self, BlockDefinitionTable, BlockFace, BlockIdx},
    chunk::{self, NeighboringChunkData, WorldgenData, CHUNK_X_SIZE, CHUNK_Y_SIZE, CHUNK_Z_SIZE},
    manager::{Manager, UpdateData},
};

// if a chunk is within this boundary it will start to render
const MIN_RENDER_RADIUS_X: i32 = 6;
const MIN_RENDER_RADIUS_Y: i32 = 6;
const MIN_RENDER_RADIUS_Z: i32 = 6;

// if a chunk is within this boundary it will stop rendering
const MAX_RENDER_RADIUS_X: i32 = 8;
const MAX_RENDER_RADIUS_Y: i32 = 8;
const MAX_RENDER_RADIUS_Z: i32 = 8;

struct Chunk {
    data: Option<Arc<Vec<BlockIdx>>>,
    // the instant at which the data started generating
    data_started_generating: Option<Instant>,
    // the instant at which data was set
    data_set_at: Option<Instant>,
    // the entity id of the mesh
    entity_id: Option<u32>,
    // the instant at which the mesh started generating
    mesh_started_generating: Option<Instant>,
    // the instant at which the mesh was set inside the scene
    mesh_set_at: Option<Instant>,
}

enum ChunkWorkerEvent {
    ChunkGenerated(Point3<i32>, Vec<BlockIdx>),
    ChunkMeshed(Point3<i32>, Instant, SceneUploadedObjectHandle, Option<Collider>),
}

struct InnerChunkManager {
    memory_allocator: Arc<dyn MemoryAllocator>,
    uploader: SceneUploader,
    threadpool: Arc<ThreadPool>,
    worldgen_data: WorldgenData,
    center_chunk: Point3<i32>,
    chunks: HashMap<Point3<i32>, Chunk>,
    event_sender: Sender<ChunkWorkerEvent>,
    event_reciever: Receiver<ChunkWorkerEvent>,
}

impl InnerChunkManager {
    pub fn new(
        uploader: SceneUploader,
        threadpool: Arc<ThreadPool>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        seed: u32,
        block_definition_table: Arc<BlockDefinitionTable>,
    ) -> Self {
        let (event_sender, event_reciever) = std::sync::mpsc::channel();
        let mut cm = Self {
            uploader,
            threadpool,
            memory_allocator,
            worldgen_data: WorldgenData {
                noise: Arc::new(OpenSimplex::new(seed)),
                block_definition_table,
            },
            center_chunk: Point3::new(0, 0, 0),
            chunks: HashMap::new(),
            event_reciever,
            event_sender,
        };
        cm.set_center_chunk(Point3::new(0, 0, 0));
        cm
    }

    // sets the center of the chunk map.
    // this will cause chunks to be generated and unloaded as needed.
    fn set_center_chunk(&mut self, chunk_position: Point3<i32>) {
        self.center_chunk = chunk_position;
        for x in -MIN_RENDER_RADIUS_X..=MIN_RENDER_RADIUS_X {
            for y in -MIN_RENDER_RADIUS_Y..=MIN_RENDER_RADIUS_Y {
                for z in -MIN_RENDER_RADIUS_Z..=MIN_RENDER_RADIUS_Z {
                    self.chunks
                        .entry(Point3::new(
                            chunk_position[0] + x,
                            chunk_position[1] + y,
                            chunk_position[2] + z,
                        ))
                        .or_insert(Chunk {
                            data: None,
                            data_started_generating: None,
                            data_set_at: None,
                            entity_id: None,
                            mesh_started_generating: None,
                            mesh_set_at: None,
                        });
                }
            }
        }
    }

    fn adjacent_chunk_positions(chunk_coords: Point3<i32>) -> [Point3<i32>; 6] {
        [
            chunk_coords + Vector3::new(-1, 0, 0),
            chunk_coords + Vector3::new(1, 0, 0),
            chunk_coords + Vector3::new(0, -1, 0),
            chunk_coords + Vector3::new(0, 1, 0),
            chunk_coords + Vector3::new(0, 0, -1),
            chunk_coords + Vector3::new(0, 0, 1),
        ]
    }

    fn adjacent_chunks<'a>(&'a self, chunk_coords: Point3<i32>) -> [Option<&'a Chunk>; 6] {
        let mut out = [None, None, None, None, None, None];
        for (i, position) in Self::adjacent_chunk_positions(chunk_coords)
            .iter()
            .enumerate()
        {
            out[i] = self.chunks.get(position);
        }
        out
    }

    fn adjacent_chunks_have_data(&self, chunk_position: Point3<i32>) -> bool {
        self.adjacent_chunks(chunk_position)
            .iter()
            .all(|x| x.is_some() && x.unwrap().data.is_some())
    }

    fn unwrap_adjacent_chunks(&self, chunk_coords: Point3<i32>) -> [Arc<Vec<BlockIdx>>; 6] {
        let adjacent_chunks: Vec<Arc<Vec<BlockIdx>>> = self
            .adjacent_chunks(chunk_coords)
            .iter()
            .map(|x| x.unwrap().data.clone().unwrap())
            .collect();
        adjacent_chunks.try_into().unwrap()
    }

    fn chunk_should_be_loaded(&self, chunk_position: Point3<i32>) -> bool {
        let distance = chunk_position - self.center_chunk;
        distance[0].abs() <= MAX_RENDER_RADIUS_X
            && distance[1].abs() <= MAX_RENDER_RADIUS_Y
            && distance[2].abs() <= MAX_RENDER_RADIUS_Z
    }

    fn update_chunks(&mut self, reserve_entity_id: &mut dyn FnMut() -> u32) -> Vec<WorldChange> {
        // get sorted chunk positions by distance from center
        let chunk_positions: Vec<Point3<i32>> = self.chunks.keys().cloned().collect();

        // chunk_positions
        //     .sort_by_key(|x| (x - self.center_chunk).cast::<f32>().norm_squared() as i32);

        let mut world_changes: Vec<WorldChange> = vec![];

        for chunk_position in chunk_positions {
            if !self.chunk_should_be_loaded(chunk_position) {
                let chunk = self.chunks.remove(&chunk_position).unwrap();
                if let Some(entity_id) = chunk.entity_id {
                    world_changes.push(WorldChange::GlobalEntityRemove(entity_id));
                }
                continue;
            }

            let chunk = self.chunks.get(&chunk_position).unwrap();

            let should_generate_data = match (&chunk.data, chunk.data_started_generating) {
                (None, None) => true,
                _ => false,
            };

            let should_mesh = chunk.data.is_some()
                && match (chunk.data_set_at, chunk.mesh_started_generating) {
                    (Some(_), None) => true,
                    (Some(data_set_at), Some(mesh_started_generating))
                        if data_set_at > mesh_started_generating =>
                    {
                        true
                    }
                    _ => false,
                }
                && self.adjacent_chunks_have_data(chunk_position);

            // begin asynchronously generating all chunks that need to be generated
            if should_generate_data {
                let worldgen_data = self.worldgen_data.clone();
                let event_sender = self.event_sender.clone();
                self.threadpool.execute(move || {
                    let chunk_data = chunk::generate_chunk(&worldgen_data, chunk_position);
                    let _ = event_sender
                        .send(ChunkWorkerEvent::ChunkGenerated(chunk_position, chunk_data));
                });
                let chunk = self.chunks.get_mut(&chunk_position).unwrap();
                chunk.data_started_generating = Some(Instant::now());
            }

            if should_mesh {
                let block_table = self.worldgen_data.block_definition_table.clone();
                let event_sender = self.event_sender.clone();
                let chunk = self.chunks.get(&chunk_position).unwrap();
                let data = chunk.data.clone().unwrap();
                let data_set_time = chunk.data_set_at.unwrap();

                let [left, right, down, up, back, front] =
                    self.unwrap_adjacent_chunks(chunk_position);

                let uploader = self.uploader.clone();
                self.threadpool.execute(move || {
                    let vertexes = chunk::gen_mesh(
                        &block_table,
                        &data,
                        NeighboringChunkData {
                            left: &left,
                            right: &right,
                            down: &down,
                            up: &up,
                            back: &back,
                            front: &front,
                        },
                    );

                    let mesh = uploader.upload_object(vertexes);

                    let hitbox = chunk::gen_hitbox(&block_table, &data);
                
                    let _ = event_sender.send(ChunkWorkerEvent::ChunkMeshed(
                        chunk_position,
                        data_set_time,
                        mesh,
                        hitbox,
                    ));
                });
                let chunk = self.chunks.get_mut(&chunk_position).unwrap();
                chunk.mesh_started_generating = Some(Instant::now());
            }
        }

        // recieve updates from worker threads
        for event in self.event_reciever.try_iter() {
            match event {
                ChunkWorkerEvent::ChunkGenerated(chunk_position, chunk_data) => {
                    if let Some(chunk) = self.chunks.get_mut(&chunk_position) {
                        chunk.data = Some(Arc::new(chunk_data));
                        chunk.data_set_at = Some(Instant::now());
                    }
                }
                ChunkWorkerEvent::ChunkMeshed(chunk_position, mesh_data_set_at, mesh, hitbox) => {
                    if let Some(Chunk {
                        data_set_at: Some(chunk_data_set_at),
                        entity_id,
                        ..
                    }) = self.chunks.get_mut(&chunk_position)
                    {
                        // chunk data set at could be newer than mesh data set at, if the chunk was
                        // modified while the mesh was being generated
                        if &*chunk_data_set_at > &mesh_data_set_at {
                            // this mesh is stale, ignore it
                            continue;
                        }

                        // get the new entity id
                        // if the chunk already has an entity id, remove it
                        let new_entity_id = if let Some(entity_id) = entity_id {
                            world_changes.push(WorldChange::GlobalEntityRemove(*entity_id));
                            *entity_id
                        } else {
                            reserve_entity_id()
                        };

                        *entity_id = Some(new_entity_id);

                        world_changes.push(WorldChange::GlobalEntityAdd(
                            new_entity_id,
                            EntityCreationData {
                                mesh,
                                isometry: Isometry3::translation(
                                    chunk_position[0] as f32 * CHUNK_X_SIZE as f32,
                                    chunk_position[1] as f32 * CHUNK_Y_SIZE as f32,
                                    chunk_position[2] as f32 * CHUNK_Z_SIZE as f32,
                                ),
                                physics: match hitbox {
                                    Some(hitbox) => Some(EntityPhysicsData {
                                        rigid_body_type: RigidBodyType::Fixed,
                                        hitbox,
                                        linvel: Vector3::zeros(),
                                        angvel: Vector3::zeros(),
                                        controlled: false,
                                        grounded: false,
                                    }),
                                    None => None,
                                },
                            },
                        ));
                    }
                }
            }
        }

        world_changes
    }

    fn get_block(&self, global_coords: &Point3<i32>) -> Option<BlockIdx> {
        let (chunk_coords, block_coords) = chunk::global_to_chunk_coords(global_coords);
        match self.chunks.get(&chunk_coords) {
            Some(Chunk {
                data: Some(ref data),
                ..
            }) => Some(data[chunk::chunk_idx2(block_coords)]),
            _ => None,
        }
    }

    fn set_block(&mut self, global_coords: &Point3<i32>, block: BlockIdx) {
        let (chunk_coords, block_coords) = chunk::global_to_chunk_coords(global_coords);

        let old_block = match self.chunks.get_mut(&chunk_coords) {
            Some(Chunk { data, .. }) => {
                if let Some(chunk_data) = data {
                    let mut chunk_data_clone = chunk_data.as_ref().clone();
                    let old_block = chunk_data_clone[chunk::chunk_idx2(block_coords)];
                    chunk_data_clone[chunk::chunk_idx2(block_coords)] = block;
                    *data = Some(Arc::new(chunk_data_clone));

                    Some(old_block)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(_) = old_block {
            self.chunks.get_mut(&chunk_coords).unwrap().data_set_at = Some(Instant::now());
            if block_coords[0] == 0 {
                if let Some(chunk) = self
                    .chunks
                    .get_mut(&(chunk_coords + Vector3::new(-1, 0, 0)))
                {
                    chunk.data_set_at = Some(Instant::now());
                }
            }
            if block_coords[0] == CHUNK_X_SIZE as i32 - 1 {
                if let Some(chunk) = self.chunks.get_mut(&(chunk_coords + Vector3::new(1, 0, 0))) {
                    chunk.data_set_at = Some(Instant::now());
                }
            }
            if block_coords[1] == 0 {
                if let Some(chunk) = self
                    .chunks
                    .get_mut(&(chunk_coords + Vector3::new(0, -1, 0)))
                {
                    chunk.data_set_at = Some(Instant::now());
                }
            }
            if block_coords[1] == CHUNK_Y_SIZE as i32 - 1 {
                if let Some(chunk) = self.chunks.get_mut(&(chunk_coords + Vector3::new(0, 1, 0))) {
                    chunk.data_set_at = Some(Instant::now());
                }
            }
            if block_coords[2] == 0 {
                if let Some(chunk) = self
                    .chunks
                    .get_mut(&(chunk_coords + Vector3::new(0, 0, -1)))
                {
                    chunk.data_set_at = Some(Instant::now());
                }
            }
            if block_coords[2] == CHUNK_Z_SIZE as i32 - 1 {
                if let Some(chunk) = self.chunks.get_mut(&(chunk_coords + Vector3::new(0, 0, 1))) {
                    chunk.data_set_at = Some(Instant::now());
                }
            }
        }
    }

    fn trace_to_solid(
        &self,
        origin: &Point3<f32>,
        direction: &Vector3<f32>,
        radius: f32,
    ) -> Option<(Point3<i32>, BlockFace)> {
        let step = 0.01;

        let direction = direction.normalize() * step;

        let mut loc = origin.clone();
        let mut loc_quantized = chunk::floor_coords(loc.into());
        loop {
            while loc_quantized == chunk::floor_coords(loc.into()) {
                loc += direction;
                if (loc - origin).norm_squared() > radius * radius {
                    return None;
                }
            }
            loc_quantized = chunk::floor_coords(loc.into());

            let block = self.get_block(&loc_quantized);
            if let Some(block) = block {
                if self.worldgen_data.block_definition_table.solid(block) {
                    let last_loc_quantized = chunk::floor_coords((loc - direction).into());
                    let delta = loc_quantized - last_loc_quantized;
                    let face = if delta[0] == -1 {
                        BlockFace::RIGHT
                    } else if delta[0] == 1 {
                        BlockFace::LEFT
                    } else if delta[1] == -1 {
                        BlockFace::UP
                    } else if delta[1] == 1 {
                        BlockFace::DOWN
                    } else if delta[2] == -1 {
                        BlockFace::FRONT
                    } else if delta[2] == 1 {
                        BlockFace::BACK
                    } else {
                        unreachable!()
                    };

                    return Some((loc_quantized, face));
                }
            } else {
                // no longer in loaded chunk
                return None;
            }
        }
    }
}

#[derive(Clone)]
pub struct ChunkQuerier {
    inner: Rc<RefCell<InnerChunkManager>>,
}

impl ChunkQuerier {
    pub fn get_block(&self, global_coords: &Point3<i32>) -> Option<BlockIdx> {
        self.inner.borrow().get_block(global_coords)
    }

    pub fn get_block_float(&self, global_coords: &Point3<f32>) -> Option<BlockIdx> {
        self.inner
            .borrow()
            .get_block(&chunk::floor_coords(*global_coords))
    }

    pub fn trace_to_solid(
        &self,
        origin: &Point3<f32>,
        direction: &Vector3<f32>,
        radius: f32,
    ) -> Option<(Point3<i32>, BlockFace)> {
        self.inner
            .borrow()
            .trace_to_solid(origin, direction, radius)
    }
}

pub struct ChunkManager {
    inner: Rc<RefCell<InnerChunkManager>>,
}

impl ChunkManager {
    pub fn new(
        threadpool: Arc<ThreadPool>,
        uploader: SceneUploader,
        memory_allocator: Arc<dyn MemoryAllocator>,
        seed: u32,
        block_definition_table: Arc<BlockDefinitionTable>,
    ) -> (Self, ChunkQuerier) {
        let inner = Rc::new(RefCell::new(InnerChunkManager::new(
            uploader,
            threadpool,
            memory_allocator,
            seed,
            block_definition_table,
        )));

        (
            Self {
                inner: inner.clone(),
            },
            ChunkQuerier { inner },
        )
    }
}

impl Manager for ChunkManager {
    fn update<'a>(&mut self, data: UpdateData<'a>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            reserve_entity_id,
            world_changes,
            ..
        } = data;

        let mut inner = self.inner.borrow_mut();

        // process updates
        for change in world_changes {
            match change {
                WorldChange::WorldSetBlock {
                    global_coords,
                    block_id,
                } => {
                    inner.set_block(global_coords, *block_id);
                }
                _ => {}
            }
        }

        let ego_location = entities
            .get(&ego_entity_id)
            .unwrap()
            .isometry
            .translation
            .vector;

        let (ego_chunk_coords, _) =
            chunk::global_to_chunk_coords(&chunk::floor_coords(ego_location.into()));

        if ego_chunk_coords != inner.center_chunk {
            inner.set_center_chunk(ego_chunk_coords);
        }

        // update chunks
        let out = inner.update_chunks(reserve_entity_id);

        out
    }
}
