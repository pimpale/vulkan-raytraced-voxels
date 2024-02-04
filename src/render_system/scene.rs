use std::{
    collections::{BTreeMap, VecDeque},
    fmt::Debug,
    sync::Arc,
};

use nalgebra::{Isometry3, Matrix3x4, Matrix4, Point3};
use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureCreateInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
        AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode, GeometryFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::graphics::vertex_input,
    sync::GpuFuture,
    DeviceSize, Packed24_8,
};

use crate::render_system::bvh::{self, aabb::Aabb};

use super::{
    bvh::BvhNode,
    vertex::{InstanceData, Vertex3D},
};

struct Object {
    isometry: Isometry3<f32>,
    vertex_buffer: Subbuffer<[Vertex3D]>,
    blas: Arc<AccelerationStructure>,
    luminance: f32,
    light_aabb: Aabb,
    light_bl_bvh_buffer: Option<Subbuffer<[BvhNode]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopLevelAccelerationStructureState {
    UpToDate,
    NeedsUpdate,
    NeedsRebuild,
}

/// Corresponds to a TLAS
pub struct Scene<K> {
    general_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    texture_luminances: Vec<f32>,
    objects: BTreeMap<K, Option<Object>>,
    // we have to keep around old objects for n_swapchain_images frames to ensure that the TLAS is not in use
    old_objects: VecDeque<Vec<Object>>,
    n_swapchain_images: usize,
    // cached data from the last frame
    cached_tlas: Option<Arc<AccelerationStructure>>,
    cached_instance_data: Option<Subbuffer<[InstanceData]>>,
    cached_luminance_bvh: Option<Subbuffer<[BvhNode]>>,
    // last frame state
    cached_tlas_state: TopLevelAccelerationStructureState,
    // command buffer all building commands are submitted to
    blas_command_buffer: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
}

#[allow(dead_code)]
impl<K> Scene<K>
where
    K: Ord + Clone + std::cmp::Eq + std::hash::Hash,
{
    pub fn new(
        general_queue: Arc<Queue>,
        transfer_queue: Arc<Queue>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        n_swapchain_images: usize,
        texture_luminances: Vec<f32>,
    ) -> Scene<K> {
        let command_buffer = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.as_ref(),
            general_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        Scene {
            general_queue,
            transfer_queue,
            command_buffer_allocator,
            memory_allocator,
            texture_luminances,
            objects: BTreeMap::new(),
            old_objects: VecDeque::from([vec![]]),
            n_swapchain_images,
            cached_tlas: None,
            cached_instance_data: None,
            cached_luminance_bvh: None,
            cached_tlas_state: TopLevelAccelerationStructureState::NeedsRebuild,
            blas_command_buffer: command_buffer,
        }
    }

    // adds a new object to the scene with the given isometry
    pub fn add_object(
        &mut self,
        key: K,
        object_handle: SceneUploadedObjectHandle,
        isometry: Isometry3<f32>,
    ) {
        match object_handle {
            SceneUploadedObjectHandle::Empty => {
                self.objects.insert(key, None);
            }
            SceneUploadedObjectHandle::Uploaded {
                vertex_buffer,
                light_bl_bvh_buffer,
                light_aabb,
                luminance,
            } => {
                let blas = create_bottom_level_acceleration_structure(
                    &mut self.blas_command_buffer,
                    self.memory_allocator.clone(),
                    &[&vertex_buffer],
                    isometry,
                );
                self.objects.insert(
                    key,
                    Some(Object {
                        isometry,
                        vertex_buffer,
                        blas,
                        luminance,
                        light_aabb,
                        light_bl_bvh_buffer,
                    }),
                );
                self.cached_tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
            }
        }
    }

    // updates the isometry of the object with the given key
    pub fn update_object(&mut self, key: K, isometry: Isometry3<f32>) {
        match self.objects.get_mut(&key) {
            Some(Some(object)) => {
                object.isometry = isometry;
                let blas = create_bottom_level_acceleration_structure(
                    &mut self.blas_command_buffer,
                    self.memory_allocator.clone(),
                    &[&object.vertex_buffer],
                    isometry,
                );
                object.blas = blas;
                if self.cached_tlas_state == TopLevelAccelerationStructureState::UpToDate {
                    self.cached_tlas_state = TopLevelAccelerationStructureState::NeedsUpdate;
                }
            }
            Some(None) => {}
            None => panic!("object with key does not exist"),
        }
    }

    pub fn remove_object(&mut self, key: K) {
        let removed = self.objects.remove(&key);
        if let Some(removed) = removed.flatten() {
            self.old_objects[0].push(removed);
            self.cached_tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
        }
    }

    // SAFETY: after calling this function, any TLAS previously returned by get_tlas() is invalid, and must not in use
    pub unsafe fn dispose_old_objects(&mut self) {
        self.old_objects.push_front(vec![]);
        while self.old_objects.len() > self.n_swapchain_images + 10 {
            self.old_objects.pop_back();
        }
    }

    // the returned TLAS may only be used after the returned future has been waited on
    pub fn get_tlas(
        &mut self,
    ) -> (
        Arc<AccelerationStructure>,
        Subbuffer<[InstanceData]>,
        Subbuffer<[BvhNode]>,
        Box<dyn GpuFuture>,
    ) {
        // rebuild the instance buffer if any object was moved, added, or removed
        if self.cached_tlas_state != TopLevelAccelerationStructureState::UpToDate {
            let instance_data = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                self.objects
                    .values()
                    .flatten()
                    .map(
                        |Object {
                             isometry,
                             vertex_buffer,
                             light_bl_bvh_buffer,
                             ..
                         }| InstanceData {
                            transform: {
                                let mat4: Matrix4<f32> = isometry.clone().into();
                                let mat3x4: Matrix3x4<f32> = mat4.fixed_view::<3, 4>(0, 0).into();
                                mat3x4.into()
                            },
                            bvh_node_buffer_addr: match light_bl_bvh_buffer {
                                Some(light_bl_bvh_buffer) => {
                                    light_bl_bvh_buffer.device_address().unwrap().get()
                                }
                                None => 0,
                            },
                            vertex_buffer_addr: vertex_buffer.device_address().unwrap().get(),
                        },
                    )
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            self.cached_instance_data = Some(instance_data);

            let ((centroids, aabbs), (luminances, instance_ids)): (
                (Vec<_>, Vec<_>),
                (Vec<_>, Vec<_>),
            ) = self
                .objects
                .values()
                .flatten()
                .enumerate()
                .filter_map(
                    |(
                        i,
                        Object {
                            light_aabb,
                            luminance,
                            isometry,
                            ..
                        },
                    )| {
                        if *luminance > 0.0 {
                            let transformed = light_aabb.transform(isometry);
                            Some((
                                (transformed.centroid(), transformed),
                                (*luminance, i as u32),
                            ))
                        } else {
                            None
                        }
                    },
                )
                .unzip();

                dbg!(&instance_ids);

            let light_tl_bvh = if centroids.len() == 0 {
                vec![BvhNode::dummy()]
            } else {
                bvh::build::build_bvh(&centroids, &aabbs, &luminances, &instance_ids)
            };

            let light_tl_bvh_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                light_tl_bvh,
            )
            .unwrap();

            self.cached_luminance_bvh = Some(light_tl_bvh_buffer);
        }

        let future = match self.cached_tlas_state {
            TopLevelAccelerationStructureState::UpToDate => {
                vulkano::sync::now(self.general_queue.device().clone()).boxed()
            }
            _ => {
                // swap command buffers
                let blas_command_buffer = std::mem::replace(
                    &mut self.blas_command_buffer,
                    AutoCommandBufferBuilder::primary(
                        self.command_buffer_allocator.as_ref(),
                        self.general_queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap(),
                );

                let blas_build_future = blas_command_buffer
                    .build()
                    .unwrap()
                    .execute(self.general_queue.clone())
                    .unwrap();

                let mut tlas_command_buffer = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.as_ref(),
                    self.general_queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // initialize tlas build
                let tlas = create_top_level_acceleration_structure(
                    &mut tlas_command_buffer,
                    self.memory_allocator.clone(),
                    &self
                        .objects
                        .values()
                        .flatten()
                        .map(|Object { blas, .. }| blas as &AccelerationStructure)
                        .collect::<Vec<_>>(),
                );

                // actually submit acceleration structure build future
                let tlas_build_future = tlas_command_buffer
                    .build()
                    .unwrap()
                    .execute_after(blas_build_future, self.general_queue.clone())
                    .unwrap();

                // update state
                self.cached_tlas = Some(tlas);

                // return the future
                tlas_build_future.boxed()
            }
        };

        // at this point the tlas is up to date
        self.cached_tlas_state = TopLevelAccelerationStructureState::UpToDate;

        // return the tlas
        return (
            self.cached_tlas.clone().unwrap(),
            self.cached_instance_data.clone().unwrap(),
            self.cached_luminance_bvh.clone().unwrap(),
            future,
        );
    }

    pub fn uploader(&self) -> SceneUploader {
        SceneUploader {
            memory_allocator: self.memory_allocator.clone(),
            texture_luminances: self.texture_luminances.clone(),
        }
    }
}

#[derive(Clone)]
pub enum SceneUploadedObjectHandle {
    Empty,
    Uploaded {
        vertex_buffer: Subbuffer<[Vertex3D]>,
        light_bl_bvh_buffer: Option<Subbuffer<[BvhNode]>>,
        light_aabb: Aabb,
        luminance: f32,
    },
}

#[derive(Clone)]
pub struct SceneUploader {
    memory_allocator: Arc<dyn MemoryAllocator>,
    texture_luminances: Vec<f32>,
}

impl SceneUploader {
    pub fn upload_object(&self, vertexes: Vec<Vertex3D>) -> SceneUploadedObjectHandle {
        if vertexes.len() == 0 {
            return SceneUploadedObjectHandle::Empty;
        }

        // check that the number of vertexes is a multiple of 3
        assert!(vertexes.len() % 3 == 0);

        // gather data for every luminous triangle
        let mut prim_centroids = vec![];
        let mut prim_aabbs = vec![];
        let mut prim_luminances = vec![];
        let mut prim_index_ids = vec![];

        for i in 0..(vertexes.len() / 3) {
            let luminance = self.texture_luminances[vertexes[i * 3 + 0].t as usize];
            if luminance > 0.0 {
                let a = Point3::from(vertexes[i * 3 + 0].position);
                let b = Point3::from(vertexes[i * 3 + 1].position);
                let c = Point3::from(vertexes[i * 3 + 2].position);
                let area = (b - a).cross(&(c - a)).norm() / 2.0;
                prim_centroids.push(Point3::from((a.coords + b.coords + c.coords) / 3.0));
                prim_aabbs.push(Aabb::from_points(&[a, b, c]));
                prim_luminances.push(luminance * area);
                prim_index_ids.push(i as u32);
            }
        }
        let vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertexes,
        )
        .unwrap();

        if prim_centroids.len() > 0 {
            let light_bl_bvh = bvh::build::build_bvh(
                &prim_centroids,
                &prim_aabbs,
                &prim_luminances,
                &prim_index_ids,
            );

            let light_aabb =
                Aabb::from_points(&[light_bl_bvh[0].min.into(), light_bl_bvh[0].max.into()]);
            let luminance = light_bl_bvh[0].luminance;

            let light_bl_bvh_buffer = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                light_bl_bvh,
            )
            .unwrap();

            SceneUploadedObjectHandle::Uploaded {
                vertex_buffer,
                light_bl_bvh_buffer: Some(light_bl_bvh_buffer),
                luminance,
                light_aabb,
            }
        } else {
            SceneUploadedObjectHandle::Uploaded {
                vertex_buffer,
                light_bl_bvh_buffer: None,
                luminance: 0.0,
                light_aabb: Aabb::Empty,
            }
        }
    }
}

fn create_top_level_acceleration_structure(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    bottom_level_acceleration_structures: &[&AccelerationStructure],
) -> Arc<AccelerationStructure> {
    let instances = bottom_level_acceleration_structures
        .iter()
        .map(
            |&bottom_level_acceleration_structure| AccelerationStructureInstance {
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
                acceleration_structure_reference: bottom_level_acceleration_structure
                    .device_address()
                    .get(),
                ..Default::default()
            },
        )
        .collect::<Vec<_>>();

    let values = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        instances,
    )
    .unwrap();

    let geometries =
        AccelerationStructureGeometries::Instances(AccelerationStructureGeometryInstancesData {
            flags: GeometryFlags::OPAQUE,
            ..AccelerationStructureGeometryInstancesData::new(
                AccelerationStructureGeometryInstancesDataType::Values(Some(values)),
            )
        });

    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let build_range_infos = [AccelerationStructureBuildRangeInfo {
        primitive_count: bottom_level_acceleration_structures.len() as _,
        primitive_offset: 0,
        first_vertex: 0,
        transform_offset: 0,
    }];

    build_acceleration_structure(
        builder,
        memory_allocator,
        AccelerationStructureType::TopLevel,
        build_info,
        &[bottom_level_acceleration_structures.len() as u32],
        build_range_infos,
    )
}

fn create_bottom_level_acceleration_structure<T: BufferContents + vertex_input::Vertex>(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    vertex_buffers: &[&Subbuffer<[T]>],
    isometry: Isometry3<f32>,
) -> Arc<AccelerationStructure> {
    let description = T::per_vertex();

    assert_eq!(description.stride, std::mem::size_of::<T>() as u32);

    let mut triangles = vec![];
    let mut max_primitive_counts = vec![];
    let mut build_range_infos = vec![];

    let isometry_matrix: [[f32; 4]; 4] = Matrix4::from(isometry).transpose().into();

    // create transform data
    let transform_data = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        [isometry_matrix[0], isometry_matrix[1], isometry_matrix[2]],
    )
    .unwrap();

    for &vertex_buffer in vertex_buffers {
        let primitive_count = vertex_buffer.len() as u32 / 3;
        triangles.push(AccelerationStructureGeometryTrianglesData {
            flags: GeometryFlags::OPAQUE,
            vertex_data: Some(vertex_buffer.clone().into_bytes()),
            vertex_stride: description.stride,
            max_vertex: vertex_buffer.len() as _,
            index_data: None,
            transform_data: Some(transform_data.clone()),
            ..AccelerationStructureGeometryTrianglesData::new(
                description.members.get("position").unwrap().format,
            )
        });
        max_primitive_counts.push(primitive_count);
        build_range_infos.push(AccelerationStructureBuildRangeInfo {
            primitive_count,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        })
    }

    let geometries = AccelerationStructureGeometries::Triangles(triangles);
    let build_info = AccelerationStructureBuildGeometryInfo {
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    build_acceleration_structure(
        builder,
        memory_allocator,
        AccelerationStructureType::BottomLevel,
        build_info,
        &max_primitive_counts,
        build_range_infos,
    )
}

fn create_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    ty: AccelerationStructureType,
    size: DeviceSize,
) -> Arc<AccelerationStructure> {
    let buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap();

    unsafe {
        AccelerationStructure::new(
            memory_allocator.device().clone(),
            AccelerationStructureCreateInfo {
                ty,
                ..AccelerationStructureCreateInfo::new(buffer)
            },
        )
        .unwrap()
    }
}

fn create_scratch_buffer(
    memory_allocator: Arc<dyn MemoryAllocator>,
    size: DeviceSize,
) -> Subbuffer<[u8]> {
    let alignment_requirement = memory_allocator
        .device()
        .physical_device()
        .properties()
        .min_acceleration_structure_scratch_offset_alignment
        .unwrap() as DeviceSize;

    let subbuffer = Buffer::new_slice::<u8>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size + alignment_requirement,
    )
    .unwrap();

    // get the next aligned offset
    let subbuffer_address: DeviceSize = subbuffer.device_address().unwrap().into();
    let aligned_offset = alignment_requirement - (subbuffer_address % alignment_requirement);

    // slice the buffer to the aligned offset
    let subbuffer2 = subbuffer.slice(aligned_offset..(aligned_offset + size));
    assert!(u64::from(subbuffer2.device_address().unwrap()) % alignment_requirement == 0);
    assert!(subbuffer2.size() == size);

    return subbuffer2;
}

// SAFETY: If build_info.geometries is AccelerationStructureGeometries::Triangles, then the data in
// build_info.geometries.triangles.vertex_data must be valid for the duration of the use of the returned
// acceleration structure.
fn build_acceleration_structure(
    builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    ty: AccelerationStructureType,
    mut build_info: AccelerationStructureBuildGeometryInfo,
    max_primitive_counts: &[u32],
    build_range_infos: impl IntoIterator<Item = AccelerationStructureBuildRangeInfo>,
) -> Arc<AccelerationStructure> {
    let device = memory_allocator.device();

    let AccelerationStructureBuildSizesInfo {
        acceleration_structure_size,
        build_scratch_size,
        ..
    } = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &build_info,
            max_primitive_counts,
        )
        .unwrap();

    let acceleration_structure =
        create_acceleration_structure(memory_allocator.clone(), ty, acceleration_structure_size);
    let scratch_buffer = create_scratch_buffer(memory_allocator.clone(), build_scratch_size);

    build_info.dst_acceleration_structure = Some(acceleration_structure.clone());
    build_info.scratch_data = Some(scratch_buffer);

    unsafe {
        builder
            .build_acceleration_structure(build_info, build_range_infos.into_iter().collect())
            .unwrap();
    }

    acceleration_structure
}
