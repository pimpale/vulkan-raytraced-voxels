use std::sync::Arc;

use nalgebra::{Isometry3, Point3, Vector3};
use noise::{NoiseFn, OpenSimplex};
use rapier3d::{
    dynamics::MassProperties,
    geometry::{Collider, ColliderBuilder, SharedShape},
};

use super::block::{BlockDefinitionTable, BlockFace, BlockIdx};
use crate::render_system::vertex::Vertex3D;

pub const CHUNK_X_SIZE: usize = 32;
pub const CHUNK_Y_SIZE: usize = 32;
pub const CHUNK_Z_SIZE: usize = 32;

pub fn chunk_idx(x: usize, y: usize, z: usize) -> usize {
    CHUNK_Z_SIZE * CHUNK_Y_SIZE * x + CHUNK_Z_SIZE * y + z
}

pub fn chunk_idx2(p: Point3<i32>) -> usize {
    chunk_idx(p.x as usize, p.y as usize, p.z as usize)
}

pub fn floor_coords(coords: Point3<f32>) -> Point3<i32> {
    Point3::new(
        coords.x.floor() as i32,
        coords.y.floor() as i32,
        coords.z.floor() as i32,
    )
}

pub fn global_to_chunk_coords(global_coords: &Point3<i32>) -> (Point3<i32>, Point3<i32>) {
    let chunk_coords = Point3::new(
        (global_coords.x as f32 / CHUNK_X_SIZE as f32).floor() as i32,
        (global_coords.y as f32 / CHUNK_Y_SIZE as f32).floor() as i32,
        (global_coords.z as f32 / CHUNK_Z_SIZE as f32).floor() as i32,
    );

    let block_coords = Point3::new(
        global_coords[0] - chunk_coords[0] * CHUNK_X_SIZE as i32,
        global_coords[1] - chunk_coords[1] * CHUNK_Y_SIZE as i32,
        global_coords[2] - chunk_coords[2] * CHUNK_Z_SIZE as i32,
    );

    (chunk_coords, block_coords)
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
                let val_here = noise.get([wx / scale1, wy / scale1, wz / scale1]) - wy / 500.0;
                let val_above = data
                    .noise
                    .get([wx / scale1, (wy + 1.0) / scale1, wz / scale1])
                    - (wy + 1.0) / 500.0;

                let thresh = 0.2;
                if val_here > thresh {
                    if val_above > thresh {
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
    // blocks[0] = 2;
    blocks
}

pub fn gen_hitbox(blocks: &BlockDefinitionTable, chunk_data: &Vec<BlockIdx>) -> Option<Collider> {
    let mut sub_colliders = vec![];

    for x in 0..CHUNK_X_SIZE {
        for y in 0..CHUNK_Y_SIZE {
            for z in 0..CHUNK_Z_SIZE {
                if blocks.solid(chunk_data[chunk_idx(x, y, z)]) {
                    let collider = SharedShape::cuboid(0.5, 0.5, 0.5);
                    let position =
                        Isometry3::translation(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    sub_colliders.push((position, collider));
                }
            }
        }
    }

    match sub_colliders.len() {
        0 => None,
        _ => {
            let collider = ColliderBuilder::compound(sub_colliders)
                // computing the mass properties is expensive, and this is terrain
                // so we can just set the mass properties to infinity
                .mass_properties(MassProperties::from_cuboid(
                    f32::INFINITY,
                    Vector3::from([
                        CHUNK_X_SIZE as f32 * 0.5,
                        CHUNK_Y_SIZE as f32 * 0.5,
                        CHUNK_Z_SIZE as f32 * 0.5,
                    ]),
                ))
                .build();

            Some(collider)
        }
    }
}

pub struct NeighboringChunkData<'a> {
    pub left: &'a Vec<BlockIdx>,
    pub right: &'a Vec<BlockIdx>,
    pub down: &'a Vec<BlockIdx>,
    pub up: &'a Vec<BlockIdx>,
    pub back: &'a Vec<BlockIdx>,
    pub front: &'a Vec<BlockIdx>,
}

pub fn gen_mesh<'a>(
    blocks: &BlockDefinitionTable,
    chunk_data: &Vec<BlockIdx>,
    neighboring_chunk_data: NeighboringChunkData<'a>,
) -> Vec<Vertex3D> {
    let mut vertexes = vec![];

    for x in 0..CHUNK_X_SIZE {
        for y in 0..CHUNK_Y_SIZE {
            for z in 0..CHUNK_Z_SIZE {
                let block_idx = chunk_data[chunk_idx(x, y, z)];
                if blocks.completely_transparent(block_idx) {
                    continue;
                }

                let left_block_idx = if x == 0 {
                    neighboring_chunk_data.left[chunk_idx(CHUNK_X_SIZE - 1, y, z)]
                } else {
                    chunk_data[chunk_idx(x - 1, y, z)]
                };

                let right_block_idx = if x == CHUNK_X_SIZE - 1 {
                    neighboring_chunk_data.right[chunk_idx(0, y, z)]
                } else {
                    chunk_data[chunk_idx(x + 1, y, z)]
                };

                let down_block_idx = if y == 0 {
                    neighboring_chunk_data.down[chunk_idx(x, CHUNK_Y_SIZE - 1, z)]
                } else {
                    chunk_data[chunk_idx(x, y - 1, z)]
                };

                let up_block_idx = if y == CHUNK_Y_SIZE - 1 {
                    neighboring_chunk_data.up[chunk_idx(x, 0, z)]
                } else {
                    chunk_data[chunk_idx(x, y + 1, z)]
                };

                let back_block_idx = if z == 0 {
                    neighboring_chunk_data.back[chunk_idx(x, y, CHUNK_Z_SIZE - 1)]
                } else {
                    chunk_data[chunk_idx(x, y, z - 1)]
                };

                let front_block_idx = if z == CHUNK_Z_SIZE - 1 {
                    neighboring_chunk_data.front[chunk_idx(x, y, 0)]
                } else {
                    chunk_data[chunk_idx(x, y, z + 1)]
                };

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
                if blocks.translucent(left_block_idx) {
                    let t = blocks.get_material_offset(block_idx, BlockFace::LEFT);
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v000, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                }

                // right face
                if blocks.translucent(right_block_idx) {
                    let t = blocks.get_material_offset(block_idx, BlockFace::RIGHT);
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
                }

                // lower face
                if blocks.translucent(down_block_idx) {
                    let t = blocks.get_material_offset(block_idx, BlockFace::DOWN);
                    vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
                }

                // upper face
                if blocks.translucent(up_block_idx) {
                    let t = blocks.get_material_offset(block_idx, BlockFace::UP);
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
                }

                // back face
                if blocks.translucent(back_block_idx) {
                    let t = blocks.get_material_offset(block_idx, BlockFace::BACK);
                    vertexes.push(Vertex3D::new2(v010, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v000, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v010, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v110, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v100, t, [1.0, 1.0]));
                }

                // front face
                if blocks.translucent(front_block_idx) {
                    let t = blocks.get_material_offset(block_idx, BlockFace::FRONT);
                    vertexes.push(Vertex3D::new2(v001, t, [1.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v101, t, [0.0, 1.0]));
                    vertexes.push(Vertex3D::new2(v111, t, [0.0, 0.0]));
                    vertexes.push(Vertex3D::new2(v011, t, [1.0, 0.0]));
                }
            }
        }
    }

    vertexes
}
