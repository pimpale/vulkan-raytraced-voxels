use nalgebra::Point2;
use winit::event::{ElementState, KeyboardInput, MouseButton, VirtualKeyCode};

#[derive(Clone, Debug)]
pub struct UserInputState1 {
    // mouse state
    pub pos: Point2<f32>,
    pub mouse_left_down: bool,
    pub mouse_right_down: bool,

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
    pub space: bool,
    pub shift: bool,
}

impl Default for UserInputState1 {
    fn default() -> Self {
        UserInputState1 {
            pos: Default::default(),
            mouse_left_down: false,
            mouse_right_down: false,
            w: false,
            a: false,
            s: false,
            d: false,
            q: false,
            e: false,
            up: false,
            left: false,
            down: false,
            right: false,
            space: false,
            shift: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct UserInputState {
    pub current: UserInputState1,
    pub previous: UserInputState1,
}

impl UserInputState {
    pub fn new() -> UserInputState {
        UserInputState {
            current: Default::default(),
            previous: Default::default(),
        }
    }

    pub fn last_key_pressed(
        input: &Vec<winit::event::WindowEvent>,
        keys: &[VirtualKeyCode],
    ) -> Option<VirtualKeyCode> {
        let mut last_key = None;
        for event in input {
            match event {
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(kc),
                            state,
                            ..
                        },
                    ..
                } => {
                    if keys.contains(kc) && state == &ElementState::Pressed {
                        last_key = Some(*kc);
                    }
                }
                _ => (),
            }
        }
        last_key
    }

    pub fn key_pressed(input: &Vec<winit::event::WindowEvent>, key: VirtualKeyCode) -> bool {
        Self::last_key_pressed(input, &[key]).is_some()
    }

    pub fn handle_input(&mut self, input: &Vec<winit::event::WindowEvent>) {
        self.previous = self.current.clone();
        let current = &mut self.current;
        for input in input {
            match input {
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    current.pos = Point2::new(position.x as f32, position.y as f32);
                }
                winit::event::WindowEvent::MouseInput { state, button, .. } => {
                    current.mouse_left_down =
                        (*state == ElementState::Pressed) && (*button == MouseButton::Left);
                    current.mouse_right_down =
                        (*state == ElementState::Pressed) && (*button == MouseButton::Right);
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
                    VirtualKeyCode::W => current.w = state == &ElementState::Pressed,
                    VirtualKeyCode::A => current.a = state == &ElementState::Pressed,
                    VirtualKeyCode::S => current.s = state == &ElementState::Pressed,
                    VirtualKeyCode::D => current.d = state == &ElementState::Pressed,
                    VirtualKeyCode::Q => current.q = state == &ElementState::Pressed,
                    VirtualKeyCode::E => current.e = state == &ElementState::Pressed,
                    VirtualKeyCode::Up => current.up = state == &ElementState::Pressed,
                    VirtualKeyCode::Left => current.left = state == &ElementState::Pressed,
                    VirtualKeyCode::Down => current.down = state == &ElementState::Pressed,
                    VirtualKeyCode::Right => current.right = state == &ElementState::Pressed,
                    VirtualKeyCode::Space => current.space = state == &ElementState::Pressed,
                    VirtualKeyCode::LShift => current.shift = state == &ElementState::Pressed,
                    _ => (),
                },
                _ => (),
            }
        }
    }
}
