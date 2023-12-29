use std::collections::HashMap;

use nalgebra::Vector3;
use rapier3d::dynamics::CCDSolver;
use rapier3d::dynamics::ImpulseJointSet;
use rapier3d::dynamics::IntegrationParameters;
use rapier3d::dynamics::IslandManager;
use rapier3d::dynamics::MultibodyJointSet;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::dynamics::RigidBodySet;
use rapier3d::geometry::BroadPhase;
use rapier3d::geometry::ColliderBuilder;
use rapier3d::geometry::ColliderSet;
use rapier3d::geometry::NarrowPhase;
use rapier3d::pipeline::PhysicsPipeline;

use crate::entity::EntityCreationData;
use crate::entity::EntityCreationPhysicsData;
use crate::entity::WorldChange;
use crate::utils;

use super::manager::Manager;
use super::manager::UpdateData;

pub struct PhysicsManager {
    // physics data
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,

    // entity data
    entities: HashMap<u32, RigidBodyHandle>,
}

impl PhysicsManager {
    pub fn new() -> PhysicsManager {
        PhysicsManager {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),

            entities: HashMap::new(),
        }
    }

    fn add_entity(&mut self, entity_id: u32, entity_creation_data: &EntityCreationData) {
        let EntityCreationData {
            physics,
            isometry,
            ..
        } = entity_creation_data;

        // add to physics solver if necessary
        if let Some(EntityCreationPhysicsData { is_dynamic, hitbox }) = physics {
            let rigid_body = match is_dynamic {
                true => RigidBodyBuilder::kinematic_velocity_based(),
                false => RigidBodyBuilder::fixed(),
            }
            .position(isometry.clone())
            .build();

            let rigid_body_handle = self.rigid_body_set.insert(rigid_body);
            self.collider_set.insert_with_parent(
                hitbox.clone(),
                rigid_body_handle,
                &mut self.rigid_body_set,
            );

            self.entities.insert(entity_id, rigid_body_handle);
        }
    }

    fn remove_entity(&mut self, entity_id: u32) {
        if let Some(rigid_body_handle) = self.entities.remove(&entity_id) {
            self.rigid_body_set.remove(
                rigid_body_handle,
                &mut self.island_manager,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                true,
            );
        }
    }
}

impl Manager for PhysicsManager {
    fn update<'a>(
        &mut self,
        data: UpdateData<'a>,
        since_last_frame: &Vec<WorldChange>,
    ) -> Vec<WorldChange> {
        // remove or add any entities that we got rid of last frame
        for world_change in since_last_frame {
            match world_change {
                WorldChange::AddEntity(entity_id, entity_creation_data) => {
                    self.add_entity(*entity_id, entity_creation_data);
                }
                WorldChange::RemoveEntity(id) => {
                    self.remove_entity(*id);
                }
                WorldChange::MoveEntity {
                    id,
                    velocity,
                    torque,
                } => {
                    if let Some(handle) = self.entities.get(id) {
                        let rigid_body = self.rigid_body_set.get_mut(*handle).unwrap();
                        rigid_body.set_linvel(*velocity, true);
                        rigid_body.set_angvel(*torque, true);
                    }
                }
                _ => {}
            }
        }

        let UpdateData { entities, .. } = data;
        // step physics
        self.physics_pipeline.step(
            &Vector3::new(0.0, -9.81, 0.0),
            &IntegrationParameters::default(),
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &(),
        );

        self.entities
            .iter()
            .filter_map(|(id, handle)| {
                let entity = entities.get(id).unwrap();
                let new_isometry = *self.rigid_body_set[*handle].position();
                if entity.isometry == new_isometry {
                    None
                } else {
                    Some(WorldChange::UpdateEntityIsometry(*id, new_isometry))
                }
            })
            .collect()
    }
}
