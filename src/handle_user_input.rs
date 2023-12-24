use nalgebra::Point2;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

#[derive(Clone, Debug)]
pub struct UserInputState {
    // mouse state
    pub pos: Point2<f32>,
    pub ppos: Point2<f32>,
    pub mouse_down: bool,

    // keyboard state
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub q: bool,
    pub e: bool,
    pub up: bool,
    pub left: bool,
    pub down: bool,
    pub right: bool,
}

impl UserInputState {
    pub fn new() -> UserInputState {
        UserInputState {
            pos: Default::default(),
            ppos: Default::default(),
            mouse_down: false,
            w: false,
            a: false,
            s: false,
            d: false,
            q: false,
            e: false,
            up: false,
            left: false,
            right: false,
            down: false,
        }
    }
    pub fn handle_input(&mut self, input: &winit::event::WindowEvent) {
        match input {
            winit::event::WindowEvent::CursorMoved { position, .. } => {
                self.ppos = self.pos;
                self.pos = Point2::new(position.x as f32, position.y as f32);
            }
            winit::event::WindowEvent::MouseInput { state, .. } => {
                self.down = *state == ElementState::Pressed;
            }
            winit::event::WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        virtual_keycode: Some(kc),
                        state,
                        ..
                    },
                ..
            } => match kc {
                VirtualKeyCode::W => self.w = state == &ElementState::Pressed,
                VirtualKeyCode::A => self.a = state == &ElementState::Pressed,
                VirtualKeyCode::S => self.s = state == &ElementState::Pressed,
                VirtualKeyCode::D => self.d = state == &ElementState::Pressed,
                VirtualKeyCode::Q => self.q = state == &ElementState::Pressed,
                VirtualKeyCode::E => self.e = state == &ElementState::Pressed,
                VirtualKeyCode::Up => self.up = state == &ElementState::Pressed,
                VirtualKeyCode::Left => self.left = state == &ElementState::Pressed,
                VirtualKeyCode::Down => self.down = state == &ElementState::Pressed,
                VirtualKeyCode::Right => self.right = state == &ElementState::Pressed,
                _ => (),
            },
            _ => (),
        }
    }
}
