use nalgebra::{Point3, Vector3};

#[derive(Clone, Copy, Debug)]
pub enum Aabb {
    Empty,
    NonEmpty { min: Point3<f32>, max: Point3<f32> },
}

impl Aabb {
    pub fn from_points(points: &[Point3<f32>]) -> Aabb {
        if points.len() == 0 {
            Aabb::Empty
        } else {
            let mut min = points[0];
            let mut max = points[0];
            for point in points {
                min = min.inf(point);
                max = max.sup(point);
            }
            Aabb::NonEmpty { min, max }
        }
    }

    pub fn union(a: &Aabb, b: &Aabb) -> Aabb {
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

    pub fn diagonal(&self) -> Vector3<f32> {
        match self {
            Aabb::Empty => Vector3::zeros(),
            Aabb::NonEmpty { min, max } => max - min,
        }
    }

    pub fn centroid(&self) -> Point3<f32> {
        match self {
            Aabb::Empty => Point3::origin(),
            Aabb::NonEmpty { min, max } => Point3::from((min.coords + max.coords) / 2.0),
        }
    }

    pub fn area(&self) -> f32 {
        match self {
            Aabb::Empty => 0.0,
            Aabb::NonEmpty { min, max } => {
                let diff = max - min;
                2.0 * (diff.x * diff.y + diff.x * diff.z + diff.y * diff.z)
            }
        }
    }

    pub fn min(&self) -> Point3<f32> {
        match self {
            Aabb::Empty => Point3::origin(),
            Aabb::NonEmpty { min, .. } => *min,
        }
    }

    pub fn max(&self) -> Point3<f32> {
        match self {
            Aabb::Empty => Point3::origin(),
            Aabb::NonEmpty { max, .. } => *max,
        }
    }
}
