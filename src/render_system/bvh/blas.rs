use nalgebra::{Point3, Vector3};

use crate::utils;

use super::super::vertex::Vertex3D;

trait Primitive {
    fn aabb(&self) -> Aabb;
    fn centroid(&self) -> Point3<f32>;
}

struct Triangle {
    v0: Point3<f32>,
    v1: Point3<f32>,
    v2: Point3<f32>,
}

impl Primitive for Triangle {
    fn aabb(&self) -> Aabb {
        let min = self.v0.inf(&self.v1).inf(&self.v2);
        let max = self.v0.sup(&self.v1).sup(&self.v2);
        Aabb::NonEmpty { min, max }
    }

    fn centroid(&self) -> Point3<f32> {
        Point3::from((self.v0.coords + self.v1.coords + self.v2.coords) / 3.0)
    }
}

#[derive(Clone, Copy, Debug)]
enum Aabb {
    Empty,
    NonEmpty { min: Point3<f32>, max: Point3<f32> },
}

impl Aabb {
    fn union(a: &Aabb, b: &Aabb) -> Aabb {
        match (a, b) {
            (Aabb::Empty, _) => *b,
            (_, Aabb::Empty) => *a,
            (
                Aabb::NonEmpty {
                    min: amin,
                    max: amax,
                },
                Aabb::NonEmpty {
                    min: bmin,
                    max: bmax,
                },
            ) => Aabb::NonEmpty {
                min: amin.inf(bmin),
                max: amax.sup(bmax),
            },
        }
    }

    fn diagonal(&self) -> Vector3<f32> {
        match self {
            Aabb::Empty => Vector3::zeros(),
            Aabb::NonEmpty { min, max } => max - min,
        }
    }

    fn area(&self) -> f32 {
        match self {
            Aabb::Empty => 0.0,
            Aabb::NonEmpty { min, max } => {
                let diff = max - min;
                2.0 * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct BuildBlasLeaf {
    first_prim_idx_idx: usize,
    prim_count: usize,
}

#[derive(Clone, Debug)]
struct BuildBlasInternalNode {
    left_child_idx: usize,
    right_child_idx: usize,
}

#[derive(Clone, Debug)]
enum BuildBlasNodeKind {
    Leaf(BuildBlasLeaf),
    InternalNode(BuildBlasInternalNode),
}

#[derive(Clone, Debug)]
struct BuildBlasNode {
    aabb: Aabb,
    kind: BuildBlasNodeKind,
}

fn blas_leaf_bounds(leaf: &BuildBlasLeaf, prim_idxs: &Vec<usize>, prim_aabbs: &Vec<Aabb>) -> Aabb {
    let mut bound = Aabb::Empty;
    for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
        let prim_aabb = prim_aabbs[prim_idxs[i]];
        bound = Aabb::union(&bound, &prim_aabb);
    }
    bound
}

fn find_best_plane(
    leaf: &BuildBlasLeaf,
    prim_idxs: &Vec<usize>,
    prim_centroids: &Vec<Point3<f32>>,
    prim_aabbs: &Vec<Aabb>,
    cost_function: &impl Fn(&Aabb, &Aabb, usize, usize) -> f32,
) -> (usize, f32) {
    const BINS: usize = 32;

    let mut best_cost = f32::MAX;
    let mut best_dimension = 0;
    let mut best_split_pos = 0.0;

    for dimension in 0..3 {
        // find the bounds over the centroids of all the primitives
        let mut bounds_min = f32::MAX;
        let mut bounds_max = f32::MIN;
        for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
            let centroid = &prim_centroids[prim_idxs[i]];
            bounds_min = bounds_min.min(centroid[dimension]);
            bounds_max = bounds_max.max(centroid[dimension]);
        }

        // the bounding box of each bin
        let mut bin_bounds = [Aabb::Empty; BINS];
        // the number of primitives in each bin
        let mut bin_primcount = [0; BINS];

        // assign each triangle to a bin
        let scale = BINS as f32 / (bounds_max - bounds_min);
        for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
            let prim_idx = prim_idxs[i];
            let prim_aabb = &prim_aabbs[prim_idx];
            let prim_centroid = &prim_centroids[prim_idx];
            let bin_idx = usize::min(
                BINS - 1,
                ((prim_centroid[dimension] - bounds_min) * scale) as usize,
            );
            bin_primcount[bin_idx] += 1;
            bin_bounds[bin_idx] = Aabb::union(&bin_bounds[bin_idx], &prim_aabb);
        }

        // there are BINS - 1 possible splits
        // 1 plane between every two bins
        let mut plane_aabb_to_left = [Aabb::Empty; BINS - 1];
        let mut plane_aabb_to_right = [Aabb::Empty; BINS - 1];
        let mut plane_primcount_to_left = [0; BINS - 1];
        let mut plane_primcount_to_right = [0; BINS - 1];

        let mut aabb_to_left = Aabb::Empty;
        let mut aabb_to_right = Aabb::Empty;
        let mut primcount_to_left = 0;
        let mut primcount_to_right = 0;

        for plane in 0..(BINS - 1) {
            primcount_to_left += bin_primcount[plane];
            plane_primcount_to_left[plane] = primcount_to_left;
            aabb_to_left = Aabb::union(&aabb_to_left, &bin_bounds[plane]);
            plane_aabb_to_left[plane] = aabb_to_left;

            primcount_to_right += bin_primcount[BINS - 1 - plane];
            plane_primcount_to_right[BINS - 2 - plane] = primcount_to_right;
            aabb_to_right = Aabb::union(&aabb_to_right, &bin_bounds[BINS - 1 - plane]);
            plane_aabb_to_right[BINS - 2 - plane] = aabb_to_right;
        }

        let scale = (bounds_max - bounds_min) / BINS as f32;

        for plane in 0..(BINS - 1) {
            let cost = cost_function(
                &plane_aabb_to_left[plane],
                &plane_aabb_to_right[plane],
                plane_primcount_to_left[plane],
                plane_primcount_to_right[plane],
            );
            dbg!(best_cost, cost);
            if cost < best_cost {
                best_cost = cost;
                best_dimension = dimension;
                best_split_pos = bounds_min + (plane as f32 + 1.0) * scale;
            }
        }
    }
    (best_dimension, best_split_pos)
}

fn subdivide(
    node_idx: usize,
    prim_idxs: &mut Vec<usize>,
    prim_aabbs: &Vec<Aabb>,
    prim_centroids: &Vec<Point3<f32>>,
    nodes: &mut Vec<BuildBlasNode>,
    cost_function: &impl Fn(&Aabb, &Aabb, usize, usize) -> f32,
) {
    match nodes[node_idx].kind {
        BuildBlasNodeKind::Leaf(ref leaf) if leaf.prim_count > 2 => {
            // get best plane to split along
            let (dimension, split_pos) =
                find_best_plane(leaf, prim_idxs, prim_centroids, prim_aabbs, cost_function);

            // partition the primitives in place by modifying prim_idxs
            let mut partitions = partition::partition(
                &mut prim_idxs
                    [leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count)],
                |&prim_idx| prim_centroids[prim_idx][dimension] < split_pos,
            );

            // If one of the subdivisions is empty then we fall back to randomly partitioning
            if partitions.0.len() == 0 || partitions.1.len() == 0 {
                dbg!("Falling back to random partitioning");
                partitions = prim_idxs
                    [leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count)]
                    .split_at_mut(leaf.prim_count / 2)
            }

            // create left child
            let left_leaf = BuildBlasLeaf {
                first_prim_idx_idx: leaf.first_prim_idx_idx,
                prim_count: partitions.0.len(),
            };

            // create right child
            let right_leaf = BuildBlasLeaf {
                first_prim_idx_idx: leaf.first_prim_idx_idx + partitions.0.len(),
                prim_count: partitions.1.len(),
            };

            // insert children
            let left_child_idx = insert_blas_leaf_node(left_leaf, nodes, prim_idxs, prim_aabbs);
            let right_child_idx = insert_blas_leaf_node(right_leaf, nodes, prim_idxs, prim_aabbs);

            // update parent
            nodes[node_idx].kind = BuildBlasNodeKind::InternalNode(BuildBlasInternalNode {
                left_child_idx,
                right_child_idx,
            });

            // recurse
            subdivide(
                left_child_idx,
                prim_idxs,
                prim_aabbs,
                prim_centroids,
                nodes,
                cost_function,
            );
            subdivide(
                right_child_idx,
                prim_idxs,
                prim_aabbs,
                prim_centroids,
                nodes,
                cost_function,
            );
        }
        _ => {}
    }
}

fn insert_blas_leaf_node(
    leaf: BuildBlasLeaf,
    nodes: &mut Vec<BuildBlasNode>,
    prim_idxs: &Vec<usize>,
    prim_aabbs: &Vec<Aabb>,
) -> usize {
    let node_idx = nodes.len();
    nodes.push(BuildBlasNode {
        aabb: blas_leaf_bounds(&leaf, prim_idxs, prim_aabbs),
        kind: BuildBlasNodeKind::Leaf(leaf),
    });
    node_idx
}

fn build_blas<T>(primitives: Vec<T>) -> Vec<BuildBlasNode>
where
    T: Primitive,
{
    let prim_centroids = primitives.iter().map(|p| p.centroid()).collect::<Vec<_>>();
    let prim_aabbs = primitives.iter().map(|p| p.aabb()).collect::<Vec<_>>();
    let mut prim_idxs = (0..primitives.len()).collect::<Vec<_>>();

    let mut nodes = vec![];

    // create root node
    let root_node_idx = insert_blas_leaf_node(
        BuildBlasLeaf {
            first_prim_idx_idx: 0,
            prim_count: primitives.len(),
        },
        &mut nodes,
        &prim_idxs,
        &prim_aabbs,
    );

    // surface area metric
    fn cost_function(aabb1: &Aabb, aabb2: &Aabb, count1: usize, count2: usize) -> f32 {
        aabb1.area() * count1 as f32 + aabb2.area() * count2 as f32
    }

    subdivide(
        root_node_idx,
        &mut prim_idxs,
        &prim_aabbs,
        &prim_centroids,
        &mut nodes,
        &cost_function,
    );

    // nodes now contains a list of all the nodes in the blas.
    // however, it contains rust constructs and is not able to be passed to the shader
    // we now need to convert it into a list of floats
    nodes
}

// creates a visualization of the blas by turning it into a mesh
fn create_blas_visualization(blas_nodes: &Vec<BuildBlasNode>) -> Vec<Vertex3D> {
    fn create_blas_visualization_inner(
        node_idx: usize,
        blas_nodes: &Vec<BuildBlasNode>,
        vertexes: &mut Vec<Vertex3D>,
    ) {
        // insert aabb into vertexes
        let aabb = blas_nodes[node_idx].aabb;
        match aabb {
            Aabb::Empty => {}
            Aabb::NonEmpty { min, max } => {
                let loc = Point3::from((min.coords + max.coords) / 2.0);
                let dims = max - min;
                vertexes.extend(utils::cuboid(loc, dims));
            }
        }

        match blas_nodes[node_idx].kind {
            BuildBlasNodeKind::Leaf(_) => {}
            BuildBlasNodeKind::InternalNode(ref internal_node) => {
                create_blas_visualization_inner(internal_node.left_child_idx, blas_nodes, vertexes);
                create_blas_visualization_inner(
                    internal_node.right_child_idx,
                    blas_nodes,
                    vertexes,
                );
            }
        }
    }

    let mut vertexes = vec![];
    create_blas_visualization_inner(0, blas_nodes, &mut vertexes);

    vertexes
}

pub fn test_blas() -> Vec<Vertex3D> {
    let mut primitives = vec![];
    for i in 0..100 {
        // find a random point
        let x = rand::random::<f32>() * 40.0 - 20.0;
        let y = rand::random::<f32>() * 40.0 - 20.0;
        let z = rand::random::<f32>() * 40.0 - 20.0;

        let v0 = Point3::new(x, y, z);
        let v1 = Point3::new(x, y + 0.1, z);
        let v2 = Point3::new(x, y, z + 0.1);

        primitives.push(Triangle { v0, v1, v2 });
    }

    let nodes = build_blas(primitives);

    create_blas_visualization(&nodes)
}
