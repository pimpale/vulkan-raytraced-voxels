use std::collections::HashMap;

use nalgebra::Point3;
use noise::OpenSimplex;

use super::{chunk::ChunkData, manager::Manager};


pub enum Chunk {
    Unloaded(ChunkData),
    Loaded { data: ChunkData, entity: u32 },
}

pub struct ChunkManager {
    pub chunks: HashMap<Point3<i32>, Chunk>,
    pub noise: OpenSimplex,
}

impl ChunkManager {
    pub fn new(seed: u32) -> ChunkManager {
        ChunkManager {
            chunks: HashMap::new(),
            noise: OpenSimplex::new(seed),
        }
    }
}

impl Manager for ChunkManager {
    fn update<'a>(
        &mut self,
        data: super::manager::UpdateData<'a>,
        since_last_step: &Vec<crate::entity::WorldChange>,
    ) -> Vec<crate::entity::WorldChange> {
        todo!()
    }
}