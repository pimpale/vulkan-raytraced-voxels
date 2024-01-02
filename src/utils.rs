use nalgebra::{Point2, Point3, Vector2, Vector3};
use rapier3d::geometry::{Collider, ColliderBuilder};

use crate::render_system::vertex::Vertex3D as Vertex;

pub fn flat_polyline(points: Vec<Vector3<f32>>, width: f32, color: [f32; 3]) -> Vec<Vertex> {
    let points: Vec<Vector3<f32>> = points
        .iter()
        .map(|p| Vector3::new(p[0], p[1], p[2]))
        .collect();
    let normals: Vec<Vector3<f32>> = std::iter::repeat([0.0, 1.0, 0.0].into())
        .take(points.len())
        .collect();
    let width: Vec<f32> = std::iter::repeat(width).take(points.len()).collect();
    let colors = std::iter::repeat(color).take(points.len() - 1).collect();
    polyline(points, normals, width, colors)
}

pub fn polyline(
    points: Vec<Vector3<f32>>,
    normals: Vec<Vector3<f32>>,
    width: Vec<f32>,
    colors: Vec<[f32; 3]>,
) -> Vec<Vertex> {
    assert!(points.len() > 1, "not enough points");
    assert!(
        points.len() == normals.len(),
        "there must be exactly one normal per point"
    );
    assert!(
        points.len() == width.len(),
        "there must be exactly one width per point"
    );
    assert!(
        points.len() - 1 == colors.len(),
        "there must be exactly one color per line segment"
    );
    // find the vector of each line segment
    let dposition_per_segment: Vec<Vector3<f32>> = points.windows(2).map(|w| w[1] - w[0]).collect();

    // dposition_per_points[0] = dposition_per_segment[0] and dposition_per_points[n] = dposition_per_segment[n-1], but it is the average of the two for the points in between
    let dposition_per_points: Vec<Vector3<f32>> = {
        let mut dposition_per_points = Vec::new();
        dposition_per_points.push(dposition_per_segment[0]);
        for i in 1..dposition_per_segment.len() {
            dposition_per_points
                .push((dposition_per_segment[i - 1] + dposition_per_segment[i]).normalize());
        }
        dposition_per_points.push(dposition_per_segment[dposition_per_segment.len() - 1]);
        dposition_per_points
    };

    // find the cross vectors (along which the width will be applied)
    let cross_vectors: Vec<Vector3<f32>> = dposition_per_points
        .iter()
        .zip(normals.iter())
        .map(|(&v, n)| v.cross(n).normalize())
        .collect();

    // find the left and right points
    let left_points: Vec<Vector3<f32>> = cross_vectors
        .iter()
        .zip(width.iter())
        .zip(points.iter())
        .map(|((v, &w), p)| p - v * w)
        .collect();

    let right_points: Vec<Vector3<f32>> = cross_vectors
        .iter()
        .zip(width.iter())
        .zip(points.iter())
        .map(|((v, &w), p)| p + v * w)
        .collect();

    let vertexes: Vec<Vertex> = std::iter::zip(left_points.windows(2), right_points.windows(2))
        .zip(colors)
        .flat_map(|((l, r), color)| {
            vec![
                Vertex::new(r[0].into(), color),
                Vertex::new(l[1].into(), color),
                Vertex::new(l[0].into(), color),
                Vertex::new(r[1].into(), color),
                Vertex::new(l[1].into(), color),
                Vertex::new(r[0].into(), color),
            ]
        })
        .collect();
    vertexes
}

pub fn cuboid(loc: Point3<f32>, dims: Vector3<f32>) -> Vec<Vertex> {
    let xsize = dims[0] * 0.5;
    let ysize = dims[1] * 0.5;
    let zsize = dims[2] * 0.5;

    let x = loc[0];
    let y = loc[1];
    let z = loc[2];

    let lbu = Vertex::new([x - xsize, y + ysize, z - zsize], [0.5, 0.9, 0.9]);
    let rbu = Vertex::new([x + xsize, y + ysize, z - zsize], [0.5, 0.5, 0.9]);
    let lfu = Vertex::new([x - xsize, y + ysize, z + zsize], [0.9, 0.5, 0.9]);
    let rfu = Vertex::new([x + xsize, y + ysize, z + zsize], [0.5, 0.9, 0.9]);
    let lbl = Vertex::new([x - xsize, y - ysize, z - zsize], [0.5, 0.5, 0.3]);
    let rbl = Vertex::new([x + xsize, y - ysize, z - zsize], [0.9, 0.5, 0.3]);
    let lfl = Vertex::new([x - xsize, y - ysize, z + zsize], [0.5, 0.5, 0.3]);
    let rfl = Vertex::new([x + xsize, y - ysize, z + zsize], [0.0, 0.0, 0.3]);

    vec![
        lbu, rbu, lfu, lfu, rfu, rbu, // upper square
        lbl, rbl, lfl, lfl, rfl, rbl, // lower square
        lfu, rfu, lfl, lfl, rfl, rfu, // front square
        lbu, rbu, lbl, lbl, rbl, rbu, // back square
        lbu, lfu, lbl, lbl, lfl, lfu, // left square
        rbu, rfu, rbl, rbl, rfl, rfu, // right square
    ]
}

pub fn unitcube() -> Vec<Vertex> {
    cuboid(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0))
}

// get axis aligned bounding box
pub fn get_aabb(obj: &[Vertex]) -> Vector3<f32> {
    let mut min = Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX);
    let mut max = Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN);
    for v in obj.iter() {
        if v.position[0] < min[0] {
            min[0] = v.position[0];
        }
        if v.position[1] < min[1] {
            min[1] = v.position[1];
        }
        if v.position[2] < min[2] {
            min[2] = v.position[2];
        }
        if v.position[0] > max[0] {
            max[0] = v.position[0];
        }
        if v.position[1] > max[1] {
            max[1] = v.position[1];
        }
        if v.position[2] > max[2] {
            max[2] = v.position[2];
        }
    }
    max - min
}

pub fn get_aabb_hitbox(obj: &[Vertex]) -> Collider {
    let dims = get_aabb(obj);
    // cuboid uses half-extents, so we divide by 2
    ColliderBuilder::cuboid(dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0).build()
}

pub fn get_normalized_mouse_coords(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let trackball_radius = extent[0].min(extent[1]) as f32;
    let center = Vector2::new(extent[0] as f32 / 2.0, extent[1] as f32 / 2.0);
    return (e - center) / trackball_radius;
}

pub fn screen_to_uv(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let x = e[0] / extent[0] as f32;
    let y = e[1] / extent[1] as f32;
    Point2::new(2.0 * x - 1.0, 2.0 * y - 1.0)
}
