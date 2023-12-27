use std::sync::Arc;

use nalgebra::Point3;
use noise::{OpenSimplex, NoiseFn};

use super::block::{BlockDefinitionTable, BlockFace, BlockIdx};
use crate::render_system::vertex::Vertex3D;

pub const CHUNK_X_SIZE: usize = 32;
pub const CHUNK_Y_SIZE: usize = 32;
pub const CHUNK_Z_SIZE: usize = 32;

pub fn chunk_idx(x: usize, y: usize, z:usize) -> usize {
   CHUNK_Z_SIZE * CHUNK_Y_SIZE * x + CHUNK_Z_SIZE * y + z
}

#[derive(Clone)]
pub struct WorldgenData {
    pub noise: Arc<OpenSimplex>,
    pub block_definition_table: Arc<BlockDefinitionTable>,
}


pub fn generate_chunk(data: &WorldgenData, chunk_position: Point3<i32>) -> Vec<BlockIdx> {
    let mut blocks: Vec<BlockIdx> = vec![0; CHUNK_X_SIZE * CHUNK_Y_SIZE * CHUNK_Z_SIZE];
    let noise = data.noise.as_ref();

    let chunk_offset = [
        chunk_position[0] * CHUNK_X_SIZE as i32,
        chunk_position[1] * CHUNK_Y_SIZE as i32,
        chunk_position[2] * CHUNK_Z_SIZE as i32,
    ];

    let air = data.block_definition_table.block_idx("air").unwrap();
    let grass = data.block_definition_table.block_idx("grass").unwrap();
    let stone = data.block_definition_table.block_idx("stone").unwrap();

    let scale1 = 20.0;
    for x in 0..CHUNK_X_SIZE {
        for y in 0..CHUNK_Y_SIZE {
            for z in 0..CHUNK_Z_SIZE {
                let xyzidx = chunk_idx(x, y, z);
                // calculate world coordinates in blocks
                let wx = x as f64 + chunk_offset[0] as f64;
                let wy = y as f64 + chunk_offset[1] as f64;
                let wz = z as f64 + chunk_offset[2] as f64;
                let val_here = noise.get([wx / scale1, wy / scale1, wz / scale1]);
                let val_above = data
                    .noise
                    .get([wx / scale1, (wy - 1.0) / scale1, wz / scale1]);

                if val_here > 0.2 {
                    if val_above > 0.2 {
                        blocks[xyzidx] = stone;
                    } else {
                        blocks[xyzidx] = grass;
                    }
                } else {
                    blocks[xyzidx] = air;
                }
            }
        }
    }
    blocks
}

pub fn gen_mesh(blocks: &BlockDefinitionTable, data: &Vec<BlockIdx>) -> Vec<Vertex3D> {
    let mut vertexes = vec![];

    for x in 0..CHUNK_X_SIZE {
        for y in 0..CHUNK_Y_SIZE {
            for z in 0..CHUNK_Z_SIZE {
                let block_idx = data[chunk_idx(x, y, z)];
                if blocks.transparent(block_idx) {
                    continue;
                }

                let fx = x as f32;
                let fy = y as f32;
                let fz = z as f32;

                let v000 = [fx + 0.0, fy + 0.0, fz + 0.0];
                let v100 = [fx + 1.0, fy + 0.0, fz + 0.0];
                let v001 = [fx + 0.0, fy + 0.0, fz + 1.0];
                let v101 = [fx + 1.0, fy + 0.0, fz + 1.0];
                let v010 = [fx + 0.0, fy + 1.0, fz + 0.0];
                let v110 = [fx + 1.0, fy + 1.0, fz + 0.0];
                let v011 = [fx + 0.0, fy + 1.0, fz + 1.0];
                let v111 = [fx + 1.0, fy + 1.0, fz + 1.0];

                // left face
                if x == 0 || blocks.transparent(data[chunk_idx(x-1, y, z)]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::LEFT);
                    vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                }

                // right face
                if x == CHUNK_X_SIZE - 1 || blocks.transparent(data[chunk_idx(x+1, y, z)]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::RIGHT);
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [1.0, 1.0]));
                }

                // upper face
                if y == 0 || blocks.transparent(data[chunk_idx(x, y-1, z)]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::UP);
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                }

                // lower face
                if y == CHUNK_Y_SIZE - 1 || blocks.transparent(data[chunk_idx(x, y+1, z)]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::DOWN);
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                }

                // back face
                if z == 0 || blocks.transparent(data[chunk_idx(x, y, z-1)]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::BACK);
                    vertexes.push(Vertex3D::new2(v000, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 1.0]));
                }

                // front face
                if z == CHUNK_Z_SIZE - 1 || blocks.transparent(data[chunk_idx(x, y, z+1)]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::FRONT);
                    vertexes.push(Vertex3D::new2(v011, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 0.0]));
                }
            }
        }
    }

    vertexes
}
