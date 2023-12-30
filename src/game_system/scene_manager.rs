use std::{cell::RefCell, rc::Rc};

use crate::{
    entity::WorldChange,
    render_system::{scene::Scene, vertex::Vertex3D},
};

use super::manager::{Manager, UpdateData};

pub struct SceneManager {
    pub scene: Rc<RefCell<Scene<u32, Vertex3D>>>,
}

impl SceneManager {
    pub fn new(scene: Rc<RefCell<Scene<u32, Vertex3D>>>) -> Self {
        Self { scene }
    }
}

impl Manager for SceneManager {
    // do nothing
    fn update<'a>(
        &mut self,
        _: UpdateData<'a>,
        since_last_frame: &Vec<WorldChange>,
    ) -> Vec<WorldChange> {
        let mut scene = self.scene.borrow_mut();
        for world_change in since_last_frame.iter() {
            match world_change {
                WorldChange::AddEntity(entity_id, entity_creation_data) => {
                    scene.add_object(
                        *entity_id,
                        &entity_creation_data.mesh,
                        entity_creation_data.isometry.clone(),
                    );
                }
                WorldChange::RemoveEntity(entity_id) => {
                    scene.remove_object(*entity_id);
                }
                WorldChange::UpdateEntityIsometry(entity_id, isometry) => {
                    scene.update_object(*entity_id, isometry.clone())
                }
                _ => {}
            }
        }

        vec![]
    }
}
