use nalgebra::{Isometry3, Point3, Vector3};

use crate::{
    render_system::bvh::{aabb::Aabb, BvhNode},
    utils,
};

use super::super::vertex::Vertex3D;

trait Primitive {
    fn aabb(&self) -> Aabb;
    fn centroid(&self) -> Point3<f32>;
    fn luminance(&self) -> f32;
}

struct Triangle {
    luminance: f32,
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

    fn luminance(&self) -> f32 {
        self.luminance
    }
}

struct BlBvh {
    nodes: Vec<BvhNode>,
    transform: Isometry3<f32>,
}

impl Primitive for BlBvh {
    fn aabb(&self) -> Aabb {
        // enumerate all corners of the bounding box:
        let min = Point3::from(self.nodes[0].min);
        let max = Point3::from(self.nodes[0].max);

        let corners = [
            self.transform * min,
            self.transform * Point3::new(min.x, min.y, max.z),
            self.transform * Point3::new(min.x, max.y, min.z),
            self.transform * Point3::new(min.x, max.y, max.z),
            self.transform * Point3::new(max.x, min.y, min.z),
            self.transform * Point3::new(max.x, min.y, max.z),
            self.transform * Point3::new(max.x, max.y, min.z),
            self.transform * max,
        ];
        Aabb::from_points(&corners)
    }

    fn centroid(&self) -> Point3<f32> {
        let min = Point3::from(self.nodes[0].min);
        let max = Point3::from(self.nodes[0].max);
        self.transform * Point3::from((min.coords + max.coords) / 2.0)
    }

    fn luminance(&self) -> f32 {
        self.nodes[0].luminance
    }
}

#[derive(Clone, Debug)]
struct BuildBvhLeaf {
    first_prim_idx_idx: usize,
    prim_count: usize,
}

#[derive(Clone, Debug)]
struct BuildBvhInternalNode {
    left_child_idx: usize,
    right_child_idx: usize,
}

#[derive(Clone, Debug)]
enum BuildBvhNodeKind {
    Leaf(BuildBvhLeaf),
    InternalNode(BuildBvhInternalNode),
}

#[derive(Clone, Debug)]
struct BuildBvhNode {
    aabb: Aabb,
    kind: BuildBvhNodeKind,
}

fn blas_leaf_bounds(leaf: &BuildBvhLeaf, prim_idxs: &[usize], prim_aabbs: &[Aabb]) -> Aabb {
    let mut bound = Aabb::Empty;
    for i in leaf.first_prim_idx_idx..(leaf.first_prim_idx_idx + leaf.prim_count) {
        let prim_aabb = prim_aabbs[prim_idxs[i]];
        bound = Aabb::union(&bound, &prim_aabb);
    }
    bound
}

fn find_best_plane(
    leaf: &BuildBvhLeaf,
    prim_idxs: &[usize],
    prim_centroids: &[Point3<f32>],
    prim_aabbs: &[Aabb],
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
    prim_idxs: &mut [usize],
    prim_aabbs: &[Aabb],
    prim_centroids: &[Point3<f32>],
    nodes: &mut Vec<BuildBvhNode>,
    cost_function: &impl Fn(&Aabb, &Aabb, usize, usize) -> f32,
) {
    match nodes[node_idx].kind {
        BuildBvhNodeKind::Leaf(ref leaf) if leaf.prim_count > 2 => {
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
            let left_leaf = BuildBvhLeaf {
                first_prim_idx_idx: leaf.first_prim_idx_idx,
                prim_count: partitions.0.len(),
            };

            // create right child
            let right_leaf = BuildBvhLeaf {
                first_prim_idx_idx: leaf.first_prim_idx_idx + partitions.0.len(),
                prim_count: partitions.1.len(),
            };

            // insert children
            let left_child_idx = insert_blas_leaf_node(left_leaf, nodes, prim_idxs, prim_aabbs);
            let right_child_idx = insert_blas_leaf_node(right_leaf, nodes, prim_idxs, prim_aabbs);

            // update parent
            nodes[node_idx].kind = BuildBvhNodeKind::InternalNode(BuildBvhInternalNode {
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
    leaf: BuildBvhLeaf,
    nodes: &mut Vec<BuildBvhNode>,
    prim_idxs: &[usize],
    prim_aabbs: &[Aabb],
) -> usize {
    let node_idx = nodes.len();
    nodes.push(BuildBvhNode {
        aabb: blas_leaf_bounds(&leaf, prim_idxs, prim_aabbs),
        kind: BuildBvhNodeKind::Leaf(leaf),
    });
    node_idx
}

pub fn build_bvh(
    // the center of each primitive
    prim_centroids: &[Point3<f32>],
    // the bounding box of each primitive
    prim_aabbs: &[Aabb],
    // how much power is in each primitive
    prim_luminances: &[f32],
    // the meaning of this parameter varies depending on whether this is a top level or bottom level bvh
    // if this is a top level bvh, then this is the instance id of each primitive
    // if this is a bottom level bvh, then this is the primitive index of each primitive
    prim_index_ids: &[u32],
) -> Vec<BvhNode> {
    let n_prims = prim_centroids.len();
    assert_eq!(n_prims, prim_aabbs.len());

    let mut prim_idxs = (0..n_prims).collect::<Vec<_>>();

    let mut nodes = vec![];

    // create root node
    let root_node_idx = insert_blas_leaf_node(
        BuildBvhLeaf {
            first_prim_idx_idx: 0,
            prim_count: n_prims,
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
    // we now need to convert it into the finalized state that is optimized for gpu consumption
    let mut opt_bvh = nodes
        .into_iter()
        .map(|node| match node.kind {
            BuildBvhNodeKind::Leaf(ref leaf) => {
                let prim_idx = prim_idxs[leaf.first_prim_idx_idx];
                BvhNode {
                    min: prim_aabbs[prim_idx].min().coords.into(),
                    max: prim_aabbs[prim_idx].max().coords.into(),
                    luminance: prim_luminances[prim_idx],
                    left_node_idx: u32::MAX,
                    right_node_idx_or_prim_idx: prim_index_ids[prim_idx] as u32,
                }
            }
            BuildBvhNodeKind::InternalNode(ref internal_node) => BvhNode {
                min: node.aabb.min().coords.into(),
                max: node.aabb.max().coords.into(),
                luminance: 0.0,
                left_node_idx: internal_node.left_child_idx as u32,
                right_node_idx_or_prim_idx: internal_node.right_child_idx as u32,
            },
        })
        .collect::<Vec<_>>();

    // compute luminance values for non-leaf nodes
    // the luminance of a node is the sum of the luminance of its children
    // the luminance of a leaf node is the luminance of the primitive it contains

    // the list is topologically sorted so we can just iterate over it in reverse order, and be sure that all the children of a node have already been processed
    for i in (0..opt_bvh.len()).rev() {
        if opt_bvh[i].left_node_idx != u32::MAX {
            // internal node
            let left_child = &opt_bvh[opt_bvh[i].left_node_idx as usize];
            let right_child = &opt_bvh[opt_bvh[i].right_node_idx_or_prim_idx as usize];
            opt_bvh[i].luminance = left_child.luminance + right_child.luminance;
        }
    }

    opt_bvh
}

// creates a visualization of the blas by turning it into a mesh
fn create_blas_visualization(blas_nodes: &Vec<BvhNode>) -> Vec<Vertex3D> {
    fn create_blas_visualization_inner(
        node_idx: usize,
        blas_nodes: &Vec<BvhNode>,
        vertexes: &mut Vec<Vertex3D>,
    ) {
        let node = &blas_nodes[node_idx];
        let loc = Point3::from((Vector3::from(node.min) + Vector3::from(node.max)) / 2.0);
        let dims = Vector3::from(node.max) - Vector3::from(node.min);
        vertexes.extend(utils::cuboid(loc, dims));

        match blas_nodes[node_idx].left_node_idx {
            u32::MAX => {}
            _ => {
                create_blas_visualization_inner(node.left_node_idx as usize, blas_nodes, vertexes);
                create_blas_visualization_inner(
                    node.right_node_idx_or_prim_idx as usize,
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
    let mut prim_aabbs = vec![];
    let mut prim_centroids = vec![];
    let mut prim_luminances = vec![];
    let mut prim_gl_ids = vec![];
    for i in 0..100 {
        // find a random point
        let x = rand::random::<f32>() * 40.0 - 20.0;
        let y = rand::random::<f32>() * 40.0 - 20.0;
        let z = rand::random::<f32>() * 40.0 - 20.0;
        let luminance = rand::random::<f32>() * 10.0;

        let v0 = Point3::new(x, y, z);
        let v1 = Point3::new(x, y + 0.1, z);
        let v2 = Point3::new(x, y, z + 0.1);

        prim_aabbs.push(Aabb::from_points(&[v0, v1, v2]));
        prim_centroids.push(Point3::from((v0.coords + v1.coords + v2.coords) / 3.0));
        prim_luminances.push(luminance);
        prim_gl_ids.push(i as u32);
    }

    let nodes = build_bvh(&prim_centroids, &prim_aabbs, &prim_luminances, &prim_gl_ids);

    create_blas_visualization(&nodes)
}
