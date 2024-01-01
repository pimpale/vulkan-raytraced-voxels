use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use nalgebra::UnitVector3;
use nalgebra::Vector3;
use rapier3d::control::EffectiveCharacterMovement;
use rapier3d::control::KinematicCharacterController;
use rapier3d::dynamics::CCDSolver;
use rapier3d::dynamics::ImpulseJointSet;
use rapier3d::dynamics::IntegrationParameters;
use rapier3d::dynamics::IslandManager;
use rapier3d::dynamics::LockedAxes;
use rapier3d::dynamics::MultibodyJointSet;
use rapier3d::dynamics::RigidBody;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::dynamics::RigidBodySet;
use rapier3d::dynamics::RigidBodyType;
use rapier3d::geometry::BroadPhase;
use rapier3d::geometry::ColliderSet;
use rapier3d::geometry::NarrowPhase;
use rapier3d::pipeline::PhysicsPipeline;
use rapier3d::pipeline::QueryFilter;
use rapier3d::pipeline::QueryPipeline;

use crate::game_system::game_world::EntityCreationData;
use crate::game_system::game_world::EntityPhysicsData;
use crate::game_system::game_world::WorldChange;

use super::manager::Manager;
use super::manager::UpdateData;

struct InnerPhysicsManager {
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
    query_pipeline: QueryPipeline,

    // entity data
    entities: HashMap<u32, RigidBodyHandle>,
}

impl InnerPhysicsManager {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),

            entities: HashMap::new(),
        }
    }

    fn add_entity(&mut self, entity_id: u32, entity_creation_data: &EntityCreationData) {
        let EntityCreationData {
            physics, isometry, ..
        } = entity_creation_data;

        // add to physics solver if necessary
        if let Some(EntityPhysicsData {
            rigid_body_type,
            hitbox,
            linvel,
            angvel,
        }) = physics
        {
            let rigid_body = match rigid_body_type {
                RigidBodyType::Dynamic => RigidBodyBuilder::dynamic(),
                RigidBodyType::Fixed => RigidBodyBuilder::fixed(),
                RigidBodyType::KinematicPositionBased => {
                    RigidBodyBuilder::kinematic_position_based()
                }
                RigidBodyType::KinematicVelocityBased => {
                    RigidBodyBuilder::kinematic_velocity_based()
                }
            }
            .position(isometry.clone())
            .linvel(linvel.clone())
            .angvel(angvel.clone())
            .locked_axes(LockedAxes::ROTATION_LOCKED_X | LockedAxes::ROTATION_LOCKED_Z)
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

    fn get_mut_entity<'a>(&'a mut self, entity_id: u32) -> Option<&'a mut RigidBody> {
        if let Some(rigid_body_handle) = self.entities.get(&entity_id) {
            self.rigid_body_set.get_mut(*rigid_body_handle)
        } else {
            None
        }
    }

    fn get_entity(&self, entity_id: u32) -> Option<RigidBody> {
        if let Some(rigid_body_handle) = self.entities.get(&entity_id) {
            Some(self.rigid_body_set[*rigid_body_handle].clone())
        } else {
            None
        }
    }

    fn update(&mut self) {
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
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );
    }

    fn move_character(&mut self, entity_id: u32, desired_translation: Vector3<f32>) -> EffectiveCharacterMovement {
        let rigid_body = self.get_entity(entity_id).unwrap();
        let collider_handle = rigid_body.colliders()[0];
        let collider = self.collider_set.get(collider_handle).unwrap();
        let kcc = KinematicCharacterController {
            up: UnitVector3::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
            offset: rapier3d::control::CharacterLength::Absolute(0.1),
            slide: true,
            autostep: None,
            max_slope_climb_angle: 0.1,
            min_slope_slide_angle: 0.2,
            snap_to_ground: None,
        };
        kcc.move_shape(
            0.01,
            &self.rigid_body_set,
            &self.collider_set,
            &self.query_pipeline,
            collider.shape(),
            rigid_body.position(),
            desired_translation,
            QueryFilter::new(),
            |_| (),
        )
    }
}

pub struct PhysicsManager {
    inner: Rc<RefCell<InnerPhysicsManager>>,
}

impl PhysicsManager {
    pub fn new() -> Self {
        let inner = Rc::new(RefCell::new(InnerPhysicsManager::new()));

        Self { inner }
    }
}

impl Manager for PhysicsManager {
    fn update<'a>(&mut self, data: UpdateData<'a>) -> Vec<WorldChange> {
        let mut inner = self.inner.borrow_mut();
        // remove or add any entities that we got rid of last frame
        for world_change in data.world_changes {
            match world_change {
                WorldChange::GlobalEntityAdd(entity_id, entity_creation_data) => {
                    inner.add_entity(*entity_id, entity_creation_data);
                }
                WorldChange::GlobalEntityRemove(id) => {
                    inner.remove_entity(*id);
                }
                WorldChange::PhysicsSetVelocity { id, linvel, angvel } => {
                    let rigid_body = inner.get_mut_entity(*id).unwrap();
                    rigid_body.set_linvel(*linvel, true);
                    rigid_body.set_angvel(*angvel, true);
                }
                WorldChange::PhysicsApplyCharacterTranslation {
                    id,
                    translation,
                    rotation,
                } => {}
                _ => {}
            }
        }

        inner.update();

        let UpdateData { entities, .. } = data;

        inner
            .entities
            .iter()
            .flat_map(|(id, handle)| {
                let entity = entities.get(id).unwrap();
                let mut changes = vec![];

                let new_isometry = *inner.rigid_body_set[*handle].position();
                if entity.isometry != new_isometry {
                    changes.push(WorldChange::GlobalEntityUpdateIsometry(*id, new_isometry));
                }
                let new_linvel = *inner.rigid_body_set[*handle].linvel();
                let new_angvel = *inner.rigid_body_set[*handle].angvel();
                if let Some(physics_data) = &entity.physics_data {
                    if physics_data.linvel != new_linvel || physics_data.angvel != new_angvel {
                        changes.push(WorldChange::GlobalEntityUpdateVelocity {
                            id: *id,
                            linvel: new_linvel,
                            angvel: new_angvel,
                        });
                    }
                }
                changes
            })
            .collect()
    }
}
