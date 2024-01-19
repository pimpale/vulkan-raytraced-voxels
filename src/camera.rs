use nalgebra::{
    Matrix, Matrix4, Point, Point2, Point3, Quaternion, UnitQuaternion, Vector2, Vector3,
};
use winit::event::{ElementState, MouseButton};

use crate::utils;

#[inline]
fn deg2rad(deg: f32) -> f32 {
    deg * std::f32::consts::PI / 180.0
}

// vectors giving the current perception of the camera
#[derive(Clone, Debug)]
struct DirVecs {
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

#[derive(Clone, Debug)]
pub struct RenderingPreferences {
    pub samples: u32, 
}

pub trait Camera {
    fn eye_front_right_up(&self) -> (Point3<f32>, Vector3<f32>, Vector3<f32>, Vector3<f32>);
    fn rendering_preferences(&self) -> RenderingPreferences;
    fn set_rendering_preferences(&mut self, prefs: RenderingPreferences);
    fn set_root_position(&mut self, pos: Point3<f32>);
    fn set_root_rotation(&mut self, rot: UnitQuaternion<f32>);
}

pub trait InteractiveCamera: Camera {
    fn update(&mut self);
    fn handle_event(&mut self, extent: [u32; 2], inputs: &Vec<winit::event::WindowEvent>);
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

    // rendering preferences
    rendering_preferences: RenderingPreferences,

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
            offset: 5.0,
            mouse_down: false,
            mouse_start: Default::default(),
            mouse_prev: Default::default(),
            mouse_curr: Default::default(),
            rendering_preferences: RenderingPreferences {
                samples: 1,
            },
        }
    }
}

impl Camera for SphericalCamera {
    // returns eye, front, right, up
    fn eye_front_right_up(&self) -> (Point3<f32>, Vector3<f32>, Vector3<f32>, Vector3<f32>) {
        let vecs = DirVecs::new(self.worldup, self.pitch, self.yaw);
        let front = self.root_rot * vecs.front;
        let right = self.root_rot * vecs.right;
        let up = self.root_rot * vecs.up;
        let eye = self.root_pos - self.offset * front;
        (eye, front, right, up)
    }

    fn rendering_preferences(&self) -> RenderingPreferences {
        self.rendering_preferences.clone()
    }

    fn set_rendering_preferences(&mut self, prefs: RenderingPreferences) {
        self.rendering_preferences = prefs;
    }

    fn set_root_position(&mut self, pos: Point3<f32>) {
        self.root_pos = pos;
    }

    fn set_root_rotation(&mut self, rot: UnitQuaternion<f32>) {
        self.root_rot = rot;
    }
}

impl InteractiveCamera for SphericalCamera {
    fn update(&mut self) {
        // do nothing
    }

    fn handle_event(&mut self, extent: [u32; 2], event: &Vec<winit::event::WindowEvent>) {
        for event in event {
            match event {
                // mouse down
                winit::event::WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Middle,
                    ..
                } => {
                    self.mouse_down = true;
                    self.mouse_start = self.mouse_curr;
                }
                // cursor move
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    self.mouse_prev = self.mouse_curr;
                    self.mouse_curr = utils::get_normalized_mouse_coords(
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
                    button: MouseButton::Middle,
                    ..
                } => {
                    self.mouse_down = false;
                }
                // scroll
                winit::event::WindowEvent::MouseWheel { delta, .. } => {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            self.offset -= 1.0 * y;
                            //if self.offset < 0.5 {
                            //    self.offset = 0.5;
                            //}
                        }
                        winit::event::MouseScrollDelta::PixelDelta(_) => {}
                    }
                }
                _ => {}
            }
        }
    }
}
