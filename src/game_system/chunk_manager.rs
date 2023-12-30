use std::{
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
use rapier3d::geometry::Collider;
use threadpool::ThreadPool;

use crate::{
    entity::{EntityCreationData, EntityCreationPhysicsData, WorldChange},
    render_system::vertex::Vertex3D,
};

use super::{
    block::{self, BlockDefinitionTable, BlockIdx},
    chunk::{self, NeighboringChunkData, WorldgenData, CHUNK_X_SIZE, CHUNK_Y_SIZE, CHUNK_Z_SIZE},
    manager::{Manager, UpdateData},
};

// if a chunk is within this boundary it will start to render
const MIN_RENDER_RADIUS_X: i32 = 5;
const MIN_RENDER_RADIUS_Y: i32 = 5;
const MIN_RENDER_RADIUS_Z: i32 = 5;

// if a chunk is within this boundary it will stop rendering
const MAX_RENDER_RADIUS_X: i32 = 8;
const MAX_RENDER_RADIUS_Y: i32 = 8;
const MAX_RENDER_RADIUS_Z: i32 = 8;

struct Chunk {
    data: Option<Arc<Vec<BlockIdx>>>,
    data_generating: bool,
    // the instant at which the mesh became stale
    mesh_stale: Option<Instant>,
    // the instant at which the mesh started generating
    mesh_generating: Option<Instant>,
    entity_id: Option<u32>,
}

enum ChunkWorkerEvent {
    ChunkGenerated(Point3<i32>, Vec<BlockIdx>),
    ChunkMeshed(Point3<i32>, Instant, Vec<Vertex3D>, Option<Collider>),
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
                            mesh_stale: None,
                            mesh_generating: None,
                            entity_id: None,
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
                    world_changes.push(WorldChange::RemoveEntity(entity_id));
                }
                continue;
            }

            let chunk = self.chunks.get(&chunk_position).unwrap();

            let should_generate_data = match (&chunk.data, chunk.data_generating) {
                (None, false) => true,
                _ => false,
            };

            let should_mesh = chunk.data.is_some()
                && match (chunk.mesh_stale, chunk.mesh_generating) {
                    (Some(_), None) => true,
                    (Some(mesh_became_stale), Some(mesh_started_generating)) => {
                        mesh_became_stale > mesh_started_generating
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
                chunk.data_generating = true;
            }

            if should_mesh {
                let block_table = self.worldgen_data.block_definition_table.clone();
                let event_sender = self.event_sender.clone();
                let chunk = self.chunks.get(&chunk_position).unwrap();
                let data = chunk.data.clone().unwrap();
                let mesh_stale_time = chunk.mesh_stale.unwrap();

                let [left, right, down, up, back, front] =
                    self.unwrap_adjacent_chunks(chunk_position);

                self.threadpool.execute(move || {
                    let mesh = chunk::gen_mesh(
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

                    //let hitbox = chunk::gen_hitbox(&block_table, &data);
                    let hitbox = None;
                    let _ = event_sender.send(ChunkWorkerEvent::ChunkMeshed(
                        chunk_position,
                        mesh_stale_time,
                        mesh,
                        hitbox,
                    ));
                });
                let chunk = self.chunks.get_mut(&chunk_position).unwrap();
                chunk.mesh_generating = Some(Instant::now());
            }
        }

        // recieve updates from worker threads
        for event in self.event_reciever.try_iter() {
            match event {
                ChunkWorkerEvent::ChunkGenerated(chunk_position, chunk_data) => {
                    if let Some(chunk) = self.chunks.get_mut(&chunk_position) {
                        chunk.data = Some(Arc::new(chunk_data));
                        chunk.data_generating = false;
                        chunk.mesh_stale = Some(Instant::now());

                        // mark all neighboring chunks as stale mesh
                        for position in Self::adjacent_chunk_positions(chunk_position) {
                            if let Some(chunk) = self.chunks.get_mut(&position) {
                                chunk.mesh_stale = Some(Instant::now());
                            }
                        }
                    }
                }
                ChunkWorkerEvent::ChunkMeshed(chunk_position, became_stale_at, mesh, hitbox) => {
                    if let Some(chunk) = self.chunks.get_mut(&chunk_position) {
                        if chunk.mesh_stale.unwrap() > became_stale_at {
                            // this mesh is stale, ignore it
                            continue;
                        }
                        chunk.mesh_stale = None;
                        chunk.mesh_generating = None;

                        // get the new entity id
                        // if the chunk already has an entity id, remove it
                        let entity_id = if let Some(entity_id) = chunk.entity_id {
                            world_changes.push(WorldChange::RemoveEntity(entity_id));
                            entity_id
                        } else {
                            reserve_entity_id()
                        };

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
                                physics: match hitbox {
                                    Some(hitbox) => Some(EntityCreationPhysicsData {
                                        is_dynamic: false,
                                        hitbox,
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

    fn get_block(&self, global_coords: Point3<i32>) -> Option<BlockIdx> {
        let (chunk_coords, block_coords) = chunk::global_to_chunk_coords(global_coords);
        match self.chunks.get(&chunk_coords) {
            Some(Chunk {
                data: Some(ref data),
                ..
            }) => Some(data[chunk::chunk_idx2(block_coords)]),
            _ => None,
        }
    }

    fn set_block(&mut self, global_coords: Point3<i32>, block: BlockIdx) {
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
            self.chunks.get_mut(&chunk_coords).unwrap().mesh_stale = Some(Instant::now());
            if block_coords[0] == 0 {
                if let Some(chunk) = self
                    .chunks
                    .get_mut(&(chunk_coords + Vector3::new(-1, 0, 0)))
                {
                    chunk.mesh_stale = Some(Instant::now());
                }
            }
            if block_coords[0] == CHUNK_X_SIZE as i32 - 1 {
                if let Some(chunk) = self.chunks.get_mut(&(chunk_coords + Vector3::new(1, 0, 0))) {
                    chunk.mesh_stale = Some(Instant::now());
                }
            }
            if block_coords[1] == 0 {
                if let Some(chunk) = self
                    .chunks
                    .get_mut(&(chunk_coords + Vector3::new(0, -1, 0)))
                {
                    chunk.mesh_stale = Some(Instant::now());
                }
            }
            if block_coords[1] == CHUNK_Y_SIZE as i32 - 1 {
                if let Some(chunk) = self.chunks.get_mut(&(chunk_coords + Vector3::new(0, 1, 0))) {
                    chunk.mesh_stale = Some(Instant::now());
                }
            }
            if block_coords[2] == 0 {
                if let Some(chunk) = self
                    .chunks
                    .get_mut(&(chunk_coords + Vector3::new(0, 0, -1)))
                {
                    chunk.mesh_stale = Some(Instant::now());
                }
            }
            if block_coords[2] == CHUNK_Z_SIZE as i32 - 1 {
                if let Some(chunk) = self.chunks.get_mut(&(chunk_coords + Vector3::new(0, 0, 1))) {
                    chunk.mesh_stale = Some(Instant::now());
                }
            }
        }
    }

    // From "A Fast Voxel Traversal Algorithm for Ray Tracing"
    // by John Amanatides and Andrew Woo, 1987
    // <http://www.cse.yorku.ca/~amana/research/grid.pdf>
    // <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.3443>
    // Extensions to the described algorithm:
    //   • Imposed a distance limit.
    //   • The face passed through to reach the current cube is provided to
    //     the callback.

    // The foundation of this algorithm is a parameterized representation of
    // the provided ray,
    //                    origin + t * direction,
    // except that t is not actually stored; rather, at any given point in the
    // traversal, we keep track of the *greater* t values which we would have
    // if we took a step sufficient to cross a cube boundary along that axis
    // (i.e. change the integer part of the coordinate) in the variables
    // tMaxX, tMaxY, and tMaxZ.
    fn trace_to_solid(
        &self,
        origin: &Point3<f32>,
        direction: &Vector3<f32>,
    ) -> Option<(Point3<i32>, block::BlockFace)> {
        // http://gamedev.stackexchange.com/questions/47362/cast-ray-to-select-block-in-voxel-game#comment160436_49423
        fn intbound(s: f32, ds: f32) -> f32 {
            if ds < 0.0 && s.floor() == s {
                return 0.0;
            }

            let ceils = if s == 0.0 { 1.0 } else { s.ceil() };

            if ds > 0.0 {
                (ceils - s) / ds
            } else {
                (s - s.floor()) / ds.abs()
            }
        }

        // Cube containing origin point.
        let mut x = origin[0].floor() as i32;
        let mut y = origin[1].floor() as i32;
        let mut z = origin[2].floor() as i32;

        // Break out direction vector.
        let step_x = direction[0].signum() as i32;
        let step_y = direction[1].signum() as i32;
        let step_z = direction[2].signum() as i32;

        // See description above. The initial values depend on the fractional
        // part of the origin.
        let mut t_max_x = intbound(origin[0], direction[0]);
        let mut t_max_y = intbound(origin[1], direction[1]);
        let mut t_max_z = intbound(origin[2], direction[2]);

        // The change in t when taking a step (always positive).
        let t_delta_x = step_x as f32 / direction[0];
        let t_delta_y = step_y as f32 / direction[1];
        let t_delta_z = step_z as f32 / direction[2];

        // Rescale from units of 1 cube-edge to units of 'direction' so we can
        // compare with 't'.
        let radius = 100.0 / direction.norm();

        // Avoids an infinite loop.
        // reject if the direction is zero
        assert!(
            !(direction[0].abs() < 0.0001
                && direction[1].abs() < 0.0001
                && direction[2].abs() < 0.0001)
        );

        let mut face = block::BlockFace::LEFT;
        loop {
            let block = self.get_block(Point3::new(x, y, z));
            if let Some(block) = block {
                if !self.worldgen_data.block_definition_table.transparent(block) {
                    // block is not transparent
                    break Some((Point3::new(x, y, z), face));
                }
            } else {
                // chunk not loaded
                break None;
            }

            // tMaxX stores the t-value at which we cross a cube boundary along the
            // X axis, and similarly for Y and Z. Therefore, choosing the least tMax
            // chooses the closest cube boundary. Only the first case of the four
            // has been commented in detail.
            if t_max_x < t_max_y {
                if t_max_x < t_max_z {
                    if t_max_x > radius {
                        break None;
                    }
                    // Update which cube we are now in.
                    x += step_x;
                    // Adjust tMaxX to the next X-oriented boundary crossing.
                    t_max_x += t_delta_x;
                    // record the normal vector of the cube face we entered.
                    face = if step_x == 1 {
                        block::BlockFace::LEFT
                    } else {
                        block::BlockFace::RIGHT
                    };
                } else {
                    if t_max_z > radius {
                        break None;
                    }
                    z += step_z;
                    t_max_z += t_delta_z;
                    face = if step_z == 1 {
                        block::BlockFace::BACK
                    } else {
                        block::BlockFace::FRONT
                    };
                }
            } else {
                if t_max_y < t_max_z {
                    if t_max_y > radius {
                        break None;
                    }
                    y += step_y;
                    t_max_y += t_delta_y;
                    face = if step_y == 1 {
                        block::BlockFace::UP
                    } else {
                        block::BlockFace::DOWN
                    };
                } else {
                    // Identical to the second case, repeated for simplicity in
                    // the conditionals.
                    if t_max_z > radius {
                        break None;
                    }
                    z += step_z;
                    t_max_z += t_delta_z;
                    face = if step_z == 1 {
                        block::BlockFace::BACK
                    } else {
                        block::BlockFace::FRONT
                    };
                }
            }
        }
    }
}

impl Manager for ChunkManager {
    fn update<'a>(&mut self, data: UpdateData<'a>, changes: &Vec<WorldChange>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            reserve_entity_id,
            ..
        } = data;

        // process updates
        for change in changes {
            match change {
                WorldChange::BreakBlock { origin, direction } => {
                    if let Some((block_coords, _)) = self.trace_to_solid(origin, direction) {
                        dbg!(&block_coords);

                        let air = self
                            .worldgen_data
                            .block_definition_table
                            .block_idx("air")
                            .unwrap();
                        self.set_block(block_coords, air);
                    }
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
            chunk::global_to_chunk_coords(chunk::floor_coords(ego_location.into()));

        if ego_chunk_coords != self.center_chunk {
            self.set_center_chunk(ego_chunk_coords);
        }

        // update chunks
        let out = self.update_chunks(reserve_entity_id);

        out
    }
}
