use std::collections::HashMap;

use nalgebra::Point3;
use noise::OpenSimplex;

use super::block::{BlockDefinitionTable, BlockFace};
use crate::render_system::vertex::Vertex3D;

pub const CHUNK_X_SIZE: usize = 16;
pub const CHUNK_Y_SIZE: usize = 16;
pub const CHUNK_Z_SIZE: usize = 16;
pub type ChunkData = [[[usize; CHUNK_Z_SIZE]; CHUNK_Y_SIZE]; CHUNK_X_SIZE];

pub fn gen_mesh(blocks: &BlockDefinitionTable, data: ChunkData) -> Vec<Vertex3D> {
    let mut vertexes = vec![];

    for x in 0..CHUNK_X_SIZE {
        for y in 0..CHUNK_Y_SIZE {
            for z in 0..CHUNK_Z_SIZE {
                let block_idx = data[x][y][z];
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
                if x == 0 || blocks.transparent(data[x - 1][y][z]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::LEFT);
                    vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                }

                // right face
                if x == CHUNK_X_SIZE - 1 || blocks.transparent(data[x + 1][y][z]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::RIGHT);
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [1.0, 1.0]));
                }

                // upper face
                if y == 0 || blocks.transparent(data[x][y - 1][z]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::UP);
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                }

                // lower face
                if y == CHUNK_Y_SIZE - 1 || blocks.transparent(data[x][y + 1][z]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::DOWN);
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                }

                // back face
                if z == 0 || blocks.transparent(data[x][y][z - 1]) {
                    let t = blocks.get_texture_offset(block_idx, BlockFace::BACK);
                    vertexes.push(Vertex3D::new2(v000, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 1.0]));
                }

                // front face
                if z == CHUNK_Z_SIZE - 1 || blocks.transparent(data[x][y][z + 1]) {
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
