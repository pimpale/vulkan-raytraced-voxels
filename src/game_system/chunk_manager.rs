use std::{
    collections::HashMap,
    rc::Rc,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
};

use nalgebra::{Isometry3, Point3};
use noise::{NoiseFn, OpenSimplex};
use threadpool::ThreadPool;

use crate::{
    entity::{EntityCreationData, EntityCreationPhysicsData, WorldChange},
    render_system::vertex::Vertex3D,
};

use super::{
    block::{BlockDefinitionTable, BlockIdx},
    chunk::{self, WorldgenData, CHUNK_X_SIZE, CHUNK_Y_SIZE, CHUNK_Z_SIZE},
    manager::{Manager, UpdateData},
};

// if a chunk is within this boundary it will start to render
const MIN_RENDER_RADIUS_X: i32 = 3;
const MIN_RENDER_RADIUS_Y: i32 = 3;
const MIN_RENDER_RADIUS_Z: i32 = 3;

// if a chunk is within this boundary it will stop rendering
const MAX_RENDER_RADIUS_X: i32 = 6;
const MAX_RENDER_RADIUS_Y: i32 = 6;
const MAX_RENDER_RADIUS_Z: i32 = 6;

struct Chunk {
    data: Option<Arc<Vec<BlockIdx>>>,
    data_generating: bool,
    mesh_stale: bool,
    mesh_generating: bool,
    entity_id: Option<u32>,
}

enum ChunkWorkerEvent {
    ChunkGenerated(Point3<i32>, Vec<BlockIdx>),
    ChunkMeshed(Point3<i32>, Vec<Vertex3D>),
}

pub struct ChunkManager {
    threadpool: Arc<ThreadPool>,
    worldgen_data: WorldgenData,
    center_chunk: Point3<i32>,
    chunks: HashMap<Point3<i32>, Chunk>,
    event_sender: Sender<ChunkWorkerEvent>,
    event_reciever: Receiver<ChunkWorkerEvent>,
}

impl ChunkManager {
    pub fn new(
        threadpool: Arc<ThreadPool>,
        seed: u32,
        block_definition_table: Arc<BlockDefinitionTable>,
    ) -> ChunkManager {
        let (event_sender, event_reciever) = std::sync::mpsc::channel();
        let mut cm = ChunkManager {
            threadpool,
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
                            data_generating: false,
                            mesh_stale: false,
                            mesh_generating: false,
                            entity_id: None,
                        });
                }
            }
        }
    }

    fn chunk_should_be_loaded(&self, chunk_position: Point3<i32>) -> bool {
        let distance = chunk_position - self.center_chunk;
        distance[0].abs() <= MAX_RENDER_RADIUS_X
            && distance[1].abs() <= MAX_RENDER_RADIUS_Y
            && distance[2].abs() <= MAX_RENDER_RADIUS_Z
    }

    fn update_chunks(&mut self, reserve_entity_id: &mut dyn FnMut() -> u32) -> Vec<WorldChange> {
        // get sorted chunk positions by distance from center
        let mut chunk_positions: Vec<Point3<i32>> = self.chunks.keys().cloned().collect();

        chunk_positions
            .sort_by_key(|x| (x - self.center_chunk).cast::<f32>().norm_squared() as i32);


        let mut world_changes: Vec<WorldChange> = vec![];


        for chunk_position in chunk_positions {
            if !self.chunk_should_be_loaded(chunk_position) {
                let chunk = self.chunks.remove(&chunk_position).unwrap();
                if let Some(entity_id) = chunk.entity_id {
                    world_changes.push(WorldChange::RemoveEntity(entity_id));
                }                
                continue;
            }

            let chunk = self.chunks.get_mut(&chunk_position).unwrap();
            // begin asynchronously generating all chunks that need to be generated
            if chunk.data.is_none() && !chunk.data_generating {
                let worldgen_data = self.worldgen_data.clone();
                let event_sender = self.event_sender.clone();
                self.threadpool.execute(move || {
                    let chunk_data = chunk::generate_chunk(&worldgen_data, chunk_position);
                    let _ = event_sender
                        .send(ChunkWorkerEvent::ChunkGenerated(chunk_position, chunk_data));
                });
                chunk.data_generating = true;
            }

            // begin asynchronously meshing all chunks that need to be meshed
            if chunk.data.is_some() && chunk.mesh_stale && !chunk.mesh_generating {
                let data = chunk.data.clone().unwrap();
                let block_table = self.worldgen_data.block_definition_table.clone();
                let event_sender = self.event_sender.clone();
                self.threadpool.execute(move || {
                    let mesh = chunk::gen_mesh(&block_table, &data);
                    let _ = event_sender.send(ChunkWorkerEvent::ChunkMeshed(chunk_position, mesh));
                });
                chunk.mesh_generating = true;
            }
        }

        // recieve updates from worker threads
        for event in self.event_reciever.try_iter() {
            match event {
                ChunkWorkerEvent::ChunkGenerated(chunk_position, chunk_data) => {
                    if let Some(chunk) = self.chunks.get_mut(&chunk_position) {
                        chunk.data = Some(Arc::new(chunk_data));
                        chunk.data_generating = false;
                        chunk.mesh_stale = true;
                    }
                }
                ChunkWorkerEvent::ChunkMeshed(chunk_position, mesh) => {
                    if let Some(chunk) = self.chunks.get_mut(&chunk_position) {
                        chunk.mesh_stale = false;
                        chunk.mesh_generating = false;

                        let entity_id = reserve_entity_id();
                        chunk.entity_id = Some(entity_id);

                        world_changes.push(WorldChange::AddEntity(
                            entity_id,
                            EntityCreationData {
                                mesh,
                                isometry: Isometry3::translation(
                                    chunk_position[0] as f32 * CHUNK_X_SIZE as f32,
                                    chunk_position[1] as f32 * CHUNK_Y_SIZE as f32,
                                    chunk_position[2] as f32 * CHUNK_Z_SIZE as f32,
                                ),
                                physics: None,
                            },
                        ));
                    }
                }
            }
        }

        world_changes
    }
}

impl Manager for ChunkManager {
    fn update<'a>(&mut self, data: UpdateData<'a>, _: &Vec<WorldChange>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            reserve_entity_id,
            ..
        } = data;
        let ego_location = entities
            .get(&ego_entity_id)
            .unwrap()
            .isometry
            .translation
            .vector;

        let ego_chunk_location = Point3::new(
            (ego_location[0] / CHUNK_X_SIZE as f32).floor() as i32,
            (ego_location[1] / CHUNK_Y_SIZE as f32).floor() as i32,
            (ego_location[2] / CHUNK_Z_SIZE as f32).floor() as i32,
        );

        if ego_chunk_location != self.center_chunk {
            self.set_center_chunk(ego_chunk_location);
        }

        // update chunks
        self.update_chunks(reserve_entity_id)
    }
}
