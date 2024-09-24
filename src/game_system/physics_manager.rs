use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use nalgebra::Vector3;

use rapier3d::dynamics::CCDSolver;
use rapier3d::dynamics::ImpulseJointSet;
use rapier3d::dynamics::IntegrationParameters;
use rapier3d::dynamics::IslandManager;
use rapier3d::dynamics::MultibodyJointSet;
use rapier3d::dynamics::RigidBody;
use rapier3d::dynamics::RigidBodyBuilder;
use rapier3d::dynamics::RigidBodyHandle;
use rapier3d::dynamics::RigidBodySet;
use rapier3d::dynamics::RigidBodyType;
use rapier3d::geometry::DefaultBroadPhase;
use rapier3d::geometry::ColliderSet;
use rapier3d::geometry::NarrowPhase;
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::pipeline::PhysicsPipeline;
use rapier3d::pipeline::QueryFilter;
use rapier3d::pipeline::QueryPipeline;

use crate::game_system::game_world::EntityCreationData;
use crate::game_system::game_world::EntityPhysicsData;
use crate::game_system::game_world::WorldChange;

use super::manager::Manager;
use super::manager::UpdateData;

struct PhysicsManagerEntityData {
    rigid_body_handle: RigidBodyHandle,
    controlled: bool,
    grounded: bool,
    clipping: bool,
}

struct InnerPhysicsManager {
    // physics data
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,

    // entity data
    entities: HashMap<u32, PhysicsManagerEntityData>,
}

impl InnerPhysicsManager {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
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
            controlled,
            grounded,
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
            .enabled_rotations(false, true, false)
            .build();

            let rigid_body_handle = self.rigid_body_set.insert(rigid_body);
            self.collider_set.insert_with_parent(
                hitbox.clone(),
                rigid_body_handle,
                &mut self.rigid_body_set,
            );

            self.entities.insert(
                entity_id,
                PhysicsManagerEntityData {
                    rigid_body_handle,
                    controlled: *controlled,
                    grounded: *grounded,
                    clipping: false,
                },
            );
        }
    }

    fn remove_entity(&mut self, entity_id: u32) {
        if let Some(PhysicsManagerEntityData {
            rigid_body_handle, ..
        }) = self.entities.remove(&entity_id)
        {
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
        if let Some(PhysicsManagerEntityData {
            rigid_body_handle, ..
        }) = self.entities.get(&entity_id)
        {
            self.rigid_body_set.get_mut(*rigid_body_handle)
        } else {
            None
        }
    }

    fn get_entity(&self, entity_id: u32) -> Option<RigidBody> {
        if let Some(PhysicsManagerEntityData {
            rigid_body_handle, ..
        }) = self.entities.get(&entity_id)
        {
            Some(self.rigid_body_set[*rigid_body_handle].clone())
        } else {
            None
        }
    }

    // shape cast and find distance to a fixed object
    fn cast_down(&self, entity_id: u32, max_distance: f32) -> (f32, bool) {
        let PhysicsManagerEntityData {
            rigid_body_handle, ..
        } = self.entities.get(&entity_id).unwrap();
        let rigidbody = self.rigid_body_set.get(*rigid_body_handle).unwrap();
        let collider_handle = rigidbody.colliders()[0];
        let collider = self.collider_set.get(collider_handle).unwrap();
        if let Some((_, hit)) = self.query_pipeline.cast_shape(
            &self.rigid_body_set,
            &self.collider_set,
            rigidbody.position(),
            &Vector3::new(0.0, -1.0, 0.0),
            collider.shape(),
            ShapeCastOptions {
                max_time_of_impact: max_distance,
                target_distance: 0.0,
                stop_at_penetration: true,
                compute_impact_geometry_on_penetration: false,
            },
            QueryFilter::only_fixed(),
        ) {
            (hit.time_of_impact, true)
        } else {
            (max_distance, false)
        }
    }

    fn update(&mut self) {
        let integration_parameters = IntegrationParameters::default();
        let gravity_y = -9.81;

        let ground_distances: HashMap<u32, (f32, bool)> = HashMap::from_iter(
            self.entities
                .iter()
                .filter(|(_, data)| {
                    data.controlled
                        && self
                            .rigid_body_set
                            .get(data.rigid_body_handle)
                            .unwrap()
                            .body_type()
                            == RigidBodyType::Dynamic
                })
                .map(|(id, _)| (*id, self.cast_down(*id, 1.0))),
        );

        for (id, data) in self.entities.iter_mut() {
            if data.controlled
                && self
                    .rigid_body_set
                    .get(data.rigid_body_handle)
                    .unwrap()
                    .body_type()
                    == RigidBodyType::Dynamic
            {
                let (dist, clipping) = ground_distances[id];

                let ground_just_below = dist < 0.05;
                let intersecting_ground = dist < 0.025;

                data.grounded = ground_just_below;
                data.clipping = clipping;

                let linvel = self.rigid_body_set[data.rigid_body_handle].linvel().clone();

                // we need to make the entity hover slightly off the ground by adding and then removing a force
                if intersecting_ground {
                    if linvel.y < 0.05 {
                        self.rigid_body_set[data.rigid_body_handle].set_linvel(
                            Vector3::new(
                                linvel.x,
                                -integration_parameters.dt * gravity_y + (0.025 - dist),
                                linvel.z,
                            ),
                            true,
                        );
                    }
                } else if ground_just_below {
                    if linvel.y < 0.00 {
                        self.rigid_body_set[data.rigid_body_handle].set_linvel(
                            Vector3::new(
                                linvel.x,
                                -integration_parameters.dt * gravity_y,
                                linvel.z,
                            ),
                            true,
                        );
                    }
                }
            }
        }

        // step physics
        self.physics_pipeline.step(
            &Vector3::new(0.0, gravity_y, 0.0),
            &integration_parameters,
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
                WorldChange::PhysicsApplyImpulse {
                    id,
                    impulse,
                    torque_impulse,
                } => {
                    let rigid_body = inner.get_mut_entity(*id).unwrap();
                    rigid_body.apply_impulse(*impulse, true);
                    rigid_body.apply_torque_impulse(*torque_impulse, true);
                }
                _ => {}
            }
        }

        let UpdateData { entities, .. } = data;

        inner.update();

        inner
            .entities
            .iter()
            .flat_map(
                |(
                    id,
                    PhysicsManagerEntityData {
                        rigid_body_handle,
                        grounded,
                        clipping,
                        ..
                    },
                )| {
                    let entity = entities.get(id).unwrap();
                    let mut changes = vec![];

                    let new_isometry = *inner.rigid_body_set[*rigid_body_handle].position();
                    if entity.isometry != new_isometry {
                        changes.push(WorldChange::GlobalEntityUpdateIsometry(*id, new_isometry));
                    }
                    let new_linvel = *inner.rigid_body_set[*rigid_body_handle].linvel();
                    let new_angvel = *inner.rigid_body_set[*rigid_body_handle].angvel();
                    if let Some(physics_data) = &entity.physics_data {
                        if physics_data.linvel != new_linvel || physics_data.angvel != new_angvel {
                            changes.push(WorldChange::GlobalEntityUpdateVelocity {
                                id: *id,
                                linvel: new_linvel,
                                angvel: new_angvel,
                            });
                        }
                        let should_be_grounded = *grounded && !*clipping;
                        if physics_data.grounded != should_be_grounded {
                            changes.push(WorldChange::GlobalEntityUpdateGroundedness(
                                *id,
                                should_be_grounded,
                            ));
                        }
                    }
                    changes
                },
            )
            .collect()
    }
}
