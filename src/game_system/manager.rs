use std::collections::{HashMap, BTreeMap};

use crate::game_system::game_world::{Entity, WorldChange};

pub struct UpdateData<'a> {
    pub entities: &'a HashMap<u32, Entity>,
    pub window_events: &'a Vec<winit::event::WindowEvent<'static>>,
    pub world_changes: &'a Vec<WorldChange>,
    pub ego_entity_id: u32,
    pub reserve_entity_id: &'a mut dyn FnMut() -> u32,
    pub extent: [u32; 2],

}

pub trait Manager {
    fn update<'a>(
        &mut self,
        data: UpdateData<'a>,
    ) -> Vec<WorldChange>;
}
