use std::sync::Arc;

use nalgebra::{Point3, Vector3};

use crate::{camera::InteractiveCamera, entity::WorldChange};

use super::manager::{Manager, UpdateData};

pub struct CameraManager {
    pub camera: Arc<Box<dyn InteractiveCamera>>,
}

impl CameraManager {
    pub fn new(camera: Arc<Box<dyn InteractiveCamera>>) -> CameraManager {
        CameraManager { camera }
    }
}

impl Manager for CameraManager {
    fn update<'a>(&mut self, data: UpdateData<'a>, _: &Vec<WorldChange>) -> Vec<WorldChange> {
        let UpdateData {
            ego_entity_id,
            entities,
            ..
        } = data;

        let ego = entities.get(&ego_entity_id).unwrap();

        // update camera
        self.camera
            .set_position(ego.isometry.translation.vector.into());
        self.camera.set_rotation(ego.isometry.rotation.into());

        // no world changes
        vec![]
    }

    fn handle_event(&mut self, extent: [u32; 2], event: &winit::event::WindowEvent) {
        self.camera.handle_event(extent, event);
    }
}
