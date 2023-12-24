use nalgebra::{Isometry3, Point3, Vector3};

use crate::render_system::vertex::Vertex3D as Vertex;

pub fn flat_polyline(points: Vec<Vector3<f32>>, width: f32, color: [f32; 4]) -> Vec<Vertex> {
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
    colors: Vec<[f32; 4]>,
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
                Vertex::new(l[0].into(), color),
                Vertex::new(l[1].into(), color),
                Vertex::new(r[0].into(), color),
                Vertex::new(r[0].into(), color),
                Vertex::new(l[1].into(), color),
                Vertex::new(r[1].into(), color),
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

    let lbu = Vertex::new([x - xsize, y + ysize, z - zsize], [0.5, 0.9, 0.9, 1.0]);
    let rbu = Vertex::new([x + xsize, y + ysize, z - zsize], [0.5, 0.5, 0.9, 1.0]);
    let lfu = Vertex::new([x - xsize, y + ysize, z + zsize], [0.9, 0.5, 0.9, 1.0]);
    let rfu = Vertex::new([x + xsize, y + ysize, z + zsize], [0.5, 0.9, 0.9, 1.0]);
    let lbl = Vertex::new([x - xsize, y - ysize, z - zsize], [0.5, 0.5, 0.3, 1.0]);
    let rbl = Vertex::new([x + xsize, y - ysize, z - zsize], [0.9, 0.5, 0.3, 1.0]);
    let lfl = Vertex::new([x - xsize, y - ysize, z + zsize], [0.5, 0.5, 0.3, 1.0]);
    let rfl = Vertex::new([x + xsize, y - ysize, z + zsize], [0.0, 0.0, 0.3, 1.0]);

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

pub fn transform(vec: &Vec<Vertex>, isometry: &Isometry3<f32>) -> Vec<Vertex> {
    vec.iter()
        .map(|v| {
            let loc: Point3<f32> = isometry * Point3::from(v.loc);
            Vertex::new(loc.into(), v.color)
        })
        .collect()
}

// get axis aligned bounding box
pub fn get_aabb(obj: &[Vertex]) -> Vector3<f32> {
    let mut min = Vector3::new(std::f32::MAX, std::f32::MAX, std::f32::MAX);
    let mut max = Vector3::new(std::f32::MIN, std::f32::MIN, std::f32::MIN);
    for v in obj.iter() {
        if v.loc[0] < min[0] {
            min[0] = v.loc[0];
        }
        if v.loc[1] < min[1] {
            min[1] = v.loc[1];
        }
        if v.loc[2] < min[2] {
            min[2] = v.loc[2];
        }
        if v.loc[0] > max[0] {
            max[0] = v.loc[0];
        }
        if v.loc[1] > max[1] {
            max[1] = v.loc[1];
        }
        if v.loc[2] > max[2] {
            max[2] = v.loc[2];
        }
    }
    max - min
}
