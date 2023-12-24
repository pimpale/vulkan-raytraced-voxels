use nalgebra::{
    Matrix, Matrix4, Point, Point2, Point3, Quaternion, UnitQuaternion, Vector2, Vector3,
};
use winit::event::ElementState;

#[inline]
fn deg2rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

// vectors giving the current perception of the camera
#[derive(Clone, Debug)]
struct DirVecs {
    // NOTE: front is actually backwards
    front: Vector3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
}

impl DirVecs {
    fn new(worldup: Vector3<f32>, pitch: f32, yaw: f32) -> DirVecs {
        let front = Vector3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize();
        // get other vectors
        let right = front.cross(&worldup).normalize();
        let up = right.cross(&front).normalize();
        // return values
        DirVecs { front, right, up }
    }
}

fn gen_perspective_projection(extent: [u32; 2]) -> Matrix4<f32> {
    let [screen_x, screen_y] = extent;
    let aspect_ratio = screen_x as f32 / screen_y as f32;
    let fov = deg2rad(90.0);
    let near = 0.1;
    let far = 100.0;
    Matrix4::new_perspective(aspect_ratio, fov, near, far)
}

// Converts a space with depth values in the range [-1, 1] to a space with depth values in the range [0, 1] 
// keeps the x and y values the same
fn vk_depth_correction() -> Matrix4<f32> {
    Matrix4::new_nonuniform_scaling(&Vector3::new(1.0, 1.0, 0.5)) * Matrix4::new_translation(&Vector3::new(0.0, 0.0, 1.0))
}

#[allow(dead_code)]
fn gen_orthographic_projection([screen_x, screen_y]: [u32; 2]) -> Matrix4<f32> {
    let scale = 100.0;
    let left = -(screen_x as f32) / scale;
    let right = screen_x as f32 / scale;
    let bottom = -(screen_y as f32) / scale;
    let top = screen_y as f32 / scale;
    vk_depth_correction() * Matrix4::new_orthographic(left, right, bottom, top, -200.0, 200.0)
}

pub trait Camera {
    fn mvp(&self, extent: [u32; 2]) -> Matrix4<f32>;
    fn set_position(&mut self, pos: Point3<f32>);
    fn set_rotation(&mut self, rot: UnitQuaternion<f32>);
}


pub trait InteractiveCamera: Camera {
    fn update(&mut self);
    fn handle_event(&mut self, extent: [u32; 2], input: &winit::event::WindowEvent);
}

fn get_normalized_mouse_coords(e: Point2<f32>, extent: [u32; 2]) -> Point2<f32> {
    let trackball_radius = extent[0].min(extent[1]) as f32;
    let center = Vector2::new(extent[0] as f32 / 2.0, extent[1] as f32 / 2.0);
    return (e - center) / trackball_radius;
}

// lets you orbit around the central point by clicking and dragging
pub struct SphericalCamera {
    // position of the camera's root point
    root_pos: Point3<f32>,
    // rotation of the camera's root point
    root_rot: UnitQuaternion<f32>,
    // world up
    worldup: Vector3<f32>,
    // offset from the root position
    offset: f32,
    // pitch
    pitch: f32,
    // yaw
    yaw: f32,

    // contains mouse data (if being dragged)
    mouse_down: bool,
    mouse_start: Point2<f32>,
    mouse_prev: Point2<f32>,
    mouse_curr: Point2<f32>,
}

impl SphericalCamera {
    pub fn new() -> SphericalCamera {
        SphericalCamera {
            root_pos: Point3::default(),
            root_rot: UnitQuaternion::identity(),
            worldup: Vector3::new(0.0, -1.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
            offset: 3.0,
            mouse_down: false,
            mouse_start: Default::default(),
            mouse_prev: Default::default(),
            mouse_curr: Default::default(),
        }
    }


}

impl Camera for SphericalCamera {
    fn mvp(&self, extent: [u32; 2]) -> Matrix4<f32> {
        let dirs = DirVecs::new(self.worldup, self.pitch, self.yaw);
        let projection = gen_perspective_projection(extent);
        let view = Matrix4::look_at_rh(&(self.root_pos - self.offset*(self.root_rot*dirs.front)), &self.root_pos, &self.worldup);
        projection * view
    }

    fn set_position(&mut self, pos: Point3<f32>) {
        self.root_pos = pos;
    }

    fn set_rotation(&mut self, rot: UnitQuaternion<f32>) {
        self.root_rot = rot;
    }
}

impl InteractiveCamera for SphericalCamera {
    fn update(&mut self) {
        // do nothing
    }

    fn handle_event(&mut self, extent: [u32; 2], event: &winit::event::WindowEvent) {
        match event {
            // mouse down
            winit::event::WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                self.mouse_down = true;
                self.mouse_start = self.mouse_curr;
            }
            // cursor move
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.mouse_prev = self.mouse_curr;
                self.mouse_curr = get_normalized_mouse_coords(
                    Point2::new(position.x as f32, position.y as f32),
                    extent,
                );
                if self.mouse_down {
                    // current and past
                    self.yaw -= (self.mouse_curr.x - self.mouse_prev.x) * 2.0;
                    self.pitch -= (self.mouse_curr.y - self.mouse_prev.y) * 2.0;

                    if self.pitch > deg2rad(89.0) {
                        self.pitch = deg2rad(89.0);
                    } else if self.pitch < -deg2rad(89.0) {
                        self.pitch = -deg2rad(89.0);
                    }
                }
            }
            // mouse up
            winit::event::WindowEvent::MouseInput {
                state: ElementState::Released,
                ..
            } => {
                self.mouse_down = false;
            }
            // scroll
            winit::event::WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        self.offset -= 0.1*y;
                        if self.offset < 0.5 {
                            self.offset = 0.5;
                        }
                        println!("offset: {}", self.offset);
                    }
                    winit::event::MouseScrollDelta::PixelDelta(_) => {}
                }
            }
            _ => {}
        }
    }
}


/// bird's eye view camera: orthographic projection, pitch of -90 degrees
pub struct BEVCamera {
        // position of the camera's root point
        root_pos: Point3<f32>,
        // rotation of the camera's root point
        root_rot: UnitQuaternion<f32>,
        // offset from the root position
        offset: f32,
}

impl BEVCamera {
    pub fn new() -> BEVCamera {
        BEVCamera {
            root_pos: Point3::default(),
            root_rot: UnitQuaternion::identity(),
            offset: 3.0,
        }
    }
}

impl Camera for BEVCamera {
    fn mvp(&self, extent: [u32; 2]) -> Matrix4<f32> {
        let front = Vector3::new(-1.0, 0.0, 0.0);
        let worldup = self.root_rot * front;
        let projection = gen_orthographic_projection(extent);
        let view = Matrix4::look_at_rh(&(self.root_pos + Vector3::new(0.0, self.offset, 0.0)), &self.root_pos, &worldup);
        projection * view
    }

    fn set_position(&mut self, pos: Point3<f32>) {
        self.root_pos = pos;
    }

    fn set_rotation(&mut self, rot: UnitQuaternion<f32>) {
        self.root_rot = rot;
    }
}

impl InteractiveCamera for BEVCamera {
    fn update(&mut self) {
        // do nothing
    }

    fn handle_event(&mut self, _extent: [u32; 2], _input: &winit::event::WindowEvent) {
        // do nothing
    }
}
