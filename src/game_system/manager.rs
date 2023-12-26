use std::collections::HashMap;

use crate::entity::{Entity, WorldChange};

pub struct UpdateData<'a> {
    pub entities: &'a HashMap<u32, Entity>,
    pub ego_entity_id: u32,
}

pub trait Manager {
    fn update<'a>(
        &mut self,
        data: UpdateData<'a>,
        since_last_step: &Vec<WorldChange>,
    ) -> Vec<WorldChange>;

    fn handle_event(&mut self, extent: [u32; 2], input: &winit::event::WindowEvent) {}
}