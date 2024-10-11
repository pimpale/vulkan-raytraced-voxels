use image::RgbaImage;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use rapier3d::geometry::{Collider, ColliderBuilder};

use crate::render_system::vertex::Vertex3D;

pub fn flat_polyline(points: Vec<Point3<f32>>, width: f32, color: [f32; 3]) -> Vec<Vertex3D> {
    let normals: Vec<Vector3<f32>> = std::iter::repeat([0.0, 1.0, 0.0].into())
        .take(points.len())
        .collect();
    let width: Vec<f32> = std::iter::repeat(width).take(points.len()).collect();
    let colors = std::iter::repeat(color).take(points.len() - 1).collect();
    polyline(points, normals, width, colors)
}

pub fn polyline(
    points: Vec<Point3<f32>>,
    normals: Vec<Vector3<f32>>,
    width: Vec<f32>,
    colors: Vec<[f32; 3]>,
) -> Vec<Vertex3D> {
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
    let left_points: Vec<Point3<f32>> = cross_vectors
        .iter()
        .zip(width.iter())
        .zip(points.iter())
        .map(|((v, &w), p)| p - v * w)
        .collect();

    let right_points: Vec<Point3<f32>> = cross_vectors
        .iter()
        .zip(width.iter())
        .zip(points.iter())
        .map(|((v, &w), p)| p + v * w)
        .collect();

    let vertexes: Vec<Vertex3D> = std::iter::zip(left_points.windows(2), right_points.windows(2))
        .zip(colors)
        .flat_map(|((l, r), color)| {
            vec![
                Vertex3D::new(r[0].into(), color),
                Vertex3D::new(l[1].into(), color),
                Vertex3D::new(l[0].into(), color),
                Vertex3D::new(r[1].into(), color),
                Vertex3D::new(l[1].into(), color),
                Vertex3D::new(r[0].into(), color),
            ]
        })
        .collect();
    vertexes
}

pub fn cuboid(loc: Point3<f32>, dims: Vector3<f32>) -> Vec<Vertex3D> {
    let fx = loc[0] - 0.5 * dims[0];
    let fy = loc[1] - 0.5 * dims[1];
    let fz = loc[2] - 0.5 * dims[2];

    let v000 = [fx + 0.0, fy + 0.0, fz + 0.0];
    let v100 = [fx + dims[0], fy + 0.0, fz + 0.0];
    let v001 = [fx + 0.0, fy + 0.0, fz + dims[2]];
    let v101 = [fx + dims[0]+0.0, fy + 0.0, fz + dims[2]];
    let v010 = [fx + 0.0, fy + dims[1], fz + 0.0];
    let v110 = [fx + dims[0], fy + dims[1], fz + 0.0];
    let v011 = [fx + 0.0, fy + dims[1], fz + dims[2]];
    let v111 = [fx + dims[0], fy + dims[1], fz + dims[2]];

    let mut vertexes = vec![];

    let off = 6*0;
    // left face
    {
        let t = 0+off;
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v000, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v011, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
    }

    // right face
    {
        let t = 1+off;
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v100, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v111, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
    }

    // lower face
    {
        let t = 2+off;
        vertexes.push(Vertex3D::new2(v000, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v001, t, [0.0, 1.0]));
    }

    // upper face
    {
        let t = 3+off;
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v010, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v111, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v110, t, [0.0, 0.0]));
    }

    // back face
    {
        let t = 4+off;
        vertexes.push(Vertex3D::new2(v010, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v000, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v010, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v110, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v100, t, [1.0, 1.0]));
    }

    // front face
    {
        let t = 5+off;
        vertexes.push(Vertex3D::new2(v001, t, [1.0, 1.0]));
        vertexes.push(Vertex3D::new2(v101, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 0.0]));
        vertexes.push(Vertex3D::new2(v101, t, [0.0, 1.0]));
        vertexes.push(Vertex3D::new2(v111, t, [0.0, 0.0]));
        vertexes.push(Vertex3D::new2(v011, t, [1.0, 0.0]));
    }

    vertexes
}

pub fn unitcube() -> Vec<Vertex3D> {
    cuboid(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0))
}

// get axis aligned bounding box
pub fn get_aabb(obj: &[Vertex3D]) -> Vector3<f32> {
    let mut min = Vector3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vector3::new(f32::MIN, f32::MIN, f32::MIN);
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

pub fn get_aabb_hitbox(obj: &[Vertex3D]) -> Collider {
    let dims = get_aabb(obj);
    // cuboid uses half-extents, so we divide by 2
    ColliderBuilder::cuboid(dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0).build()
}

pub fn get_normalized_mouse_coords(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let trackball_radius = extent[0].min(extent[1]) as f32;
    let center = Vector2::new(extent[0] as f32 / 2.0, extent[1] as f32 / 2.0);
    (e - center) / trackball_radius
}

pub fn screen_to_uv(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let x = e[0] / extent[0] as f32;
    let y = e[1] / extent[1] as f32;
    Point2::new(2.0 * x - 1.0, 2.0 * y - 1.0)
}

pub fn get_texture_luminances(texture_atlas: &Vec<(RgbaImage, RgbaImage, RgbaImage)>) -> Vec<f32> {
    texture_atlas
        .iter()
        .map(|(_, emissivity, _)| {
            let mut luminance = 0.0;
            for pixel in emissivity.pixels() {
                let [r, g, b, _] = pixel.0;
                luminance += r as f32 + g as f32 + b as f32;
            }
            luminance / (emissivity.width() * emissivity.height()) as f32
        })
        .collect()
}
