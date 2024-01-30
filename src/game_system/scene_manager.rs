use std::{cell::RefCell, rc::Rc};

use crate::{
    game_system::game_world::WorldChange,
    render_system::scene::Scene,
};

use super::manager::{Manager, UpdateData};

pub struct SceneManager {
    pub scene: Rc<RefCell<Scene<u32>>>,
}

impl SceneManager {
    pub fn new(scene: Rc<RefCell<Scene<u32>>>) -> Self {
        Self { scene }
    }
}

impl Manager for SceneManager {
    // do nothing
    fn update<'a>(&mut self, data: UpdateData<'a>) -> Vec<WorldChange> {
        let mut scene = self.scene.borrow_mut();
        for world_change in data.world_changes.iter() {
            match world_change {
                WorldChange::GlobalEntityAdd(entity_id, entity_creation_data) => {
                    scene.add_object(
                        *entity_id,
                        entity_creation_data.mesh.clone(),
                        entity_creation_data.isometry.clone().cast(),
                    );
                }
                WorldChange::GlobalEntityRemove(entity_id) => {
                    scene.remove_object(*entity_id);
                }
                WorldChange::GlobalEntityUpdateIsometry(entity_id, isometry) => {
                    scene.update_object(*entity_id, isometry.clone().cast())
                }
                _ => {}
            }
        }

        vec![]
    }
}
