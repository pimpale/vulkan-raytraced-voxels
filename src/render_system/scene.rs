use std::{collections::HashMap, sync::Arc};

use nalgebra::Isometry3;
use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildSizesInfo,
        AccelerationStructureBuildType, AccelerationStructureCreateInfo,
        AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
        AccelerationStructureInstance, AccelerationStructureType, BuildAccelerationStructureFlags,
        BuildAccelerationStructureMode, GeometryFlags, GeometryInstanceFlags,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfoTyped, PrimaryCommandBufferAbstract,
    },
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::graphics::vertex_input,
    sync::{future::FenceSignalFuture, GpuFuture},
    DeviceSize, Packed24_8,
};

use crate::object;

pub struct Object<Vertex> {
    object: Vec<Vertex>,
    isometry: Isometry3<f32>,
    vertex_buffer: Subbuffer<[Vertex]>,
    blas: Arc<AccelerationStructure>,
    blas_build_future: FenceSignalFuture<Box<dyn GpuFuture>>,
}

enum TopLevelAccelerationStructureState {
    UpToDate,
    NeedsUpdate,
    NeedsRebuild,
}

/// Corresponds to a TLAS
pub struct Scene<K, Vertex> {
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: HashMap<K, Object<Vertex>>,
    top_level_vertex_buffer: Option<Subbuffer<[Vertex]>>,
    tlas: Arc<AccelerationStructure>,
    tlas_state: TopLevelAccelerationStructureState,
}

#[allow(dead_code)]
impl<K, Vertex> Scene<K, Vertex>
where
    Vertex: vertex_input::Vertex + Clone + BufferContents,
    K: Clone + std::cmp::Eq + std::hash::Hash,
{
    pub fn new(
        queue: Arc<Queue>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> Scene<K, Vertex> {
        // assert that the vertex type must have a field called position
        assert!(Vertex::per_vertex().members.contains_key("position"));

        let (tlas, tlas_build_future) = create_top_level_acceleration_structure(
            memory_allocator.clone(),
            &command_buffer_allocator,
            queue.clone(),
            &[],
        );

        // await build
        tlas_build_future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Scene {
            queue,
            command_buffer_allocator,
            memory_allocator,
            objects: HashMap::new(),
            tlas,
            top_level_vertex_buffer: None,
            tlas_state: TopLevelAccelerationStructureState::UpToDate,
        }
    }

    // adds a new object to the scene with the given isometry
    pub fn add_object(&mut self, key: K, object: Vec<Vertex>, isometry: Isometry3<f32>) {
        let vertex_buffer = blas_vertex_buffer(self.memory_allocator.clone(), [&object]);
        let (blas, blas_build_future) = create_bottom_level_acceleration_structure(
            self.memory_allocator.clone(),
            &self.command_buffer_allocator,
            self.queue.clone(),
            &[&vertex_buffer],
        );
        self.objects.insert(
            key,
            Object {
                object,
                isometry,
                vertex_buffer,
                blas,
                blas_build_future: blas_build_future.then_signal_fence_and_flush().unwrap(),
            },
        );
        self.tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
    }

    // updates the isometry of the object with the given key
    pub fn update_object(&mut self, key: K, isometry: Isometry3<f32>) {
        // let vertex_buffer = blas_vertex_buffer(self.memory_allocator.clone(), [&object]);
        // let (blas, blas_build_future) = create_bottom_level_acceleration_structure(
        //     self.memory_allocator.clone(),
        //     &self.command_buffer_allocator,
        //     self.queue.clone(),
        //     &[&vertex_buffer],
        // );
        // self.objects.insert(
        //     key,
        //     Object {
        //         object,
        //         isometry,
        //         vertex_buffer,
        //         blas,
        //         blas_build_future,
        //     },
        // );
        // self.tlas_state = TopLevelAccelerationStructureState::NeedsUpdate;
    }

    pub fn remove_object(&mut self, key: K) {
        let removed = self.objects.remove(&key);
        if removed.is_some() {
            self.tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
        }
    }

    pub fn top_level_acceleration_structure(&mut self) -> Arc<AccelerationStructure> {
        match self.tlas_state {
            TopLevelAccelerationStructureState::UpToDate => {}
            TopLevelAccelerationStructureState::NeedsUpdate => todo!(),
            TopLevelAccelerationStructureState::NeedsRebuild => {
                // start building the tlas vertex buffer
                let (top_level_vertex_buffer, top_level_vertex_buffer_future) = tlas_vertex_buffer(
                    self.queue.clone(),
                    self.memory_allocator.clone(),
                    &self.command_buffer_allocator,
                    self.objects
                        .values()
                        .map(|Object { vertex_buffer, .. }| vertex_buffer)
                        .cloned()
                        .collect(),
                );

                // actually submit future
                let top_level_vertex_buffer_future = top_level_vertex_buffer_future
                    .then_signal_fence_and_flush()
                    .unwrap();

                // await all the blas futures on the cpu
                for object in self.objects.values() {
                    object.blas_build_future.wait(None).unwrap();
                }

                let (tlas, tlas_build_future) = create_top_level_acceleration_structure(
                    self.memory_allocator.clone(),
                    &self.command_buffer_allocator,
                    self.queue.clone(),
                    &self
                        .objects
                        .values()
                        .map(|Object { blas, .. }| blas as &AccelerationStructure)
                        .collect::<Vec<_>>(),
                );

                // actually submit acceleration structure build future
                let tlas_build_future = tlas_build_future.then_signal_fence_and_flush().unwrap();

                // await both futures
                top_level_vertex_buffer_future.wait(None).unwrap();
                tlas_build_future.wait(None).unwrap();

                // update state
                self.tlas = tlas;
                self.top_level_vertex_buffer = Some(top_level_vertex_buffer);
                self.tlas_state = TopLevelAccelerationStructureState::UpToDate;
            }
        }
        return self.tlas.clone();
    }
}

fn blas_vertex_buffer<'a, Vertex, Container>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: Container,
) -> Subbuffer<[Vertex]>
where
    Container: IntoIterator<Item = &'a Vec<Vertex>>,
    Vertex: Clone + BufferContents,
{
    let vertexes = objects
        .into_iter()
        .flatten()
        .cloned()
        .collect::<Vec<Vertex>>();

    Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY
                | BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertexes,
    )
    .unwrap()
}

// constructs a vertex buffer consisting of all the vertexes in the blas_vertex_buffers
// returns the vertex buffer + a future of when the copy is complete
fn tlas_vertex_buffer<Vertex>(
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    blas_vertex_buffers: Vec<Subbuffer<[Vertex]>>,
) -> (Subbuffer<[Vertex]>, Box<dyn GpuFuture>)
where
    Vertex: Clone + BufferContents,
{
    // create one time command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let dest_buffer = Buffer::new_slice::<Vertex>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        blas_vertex_buffers.iter().map(|buffer| buffer.len()).sum(),
    )
    .unwrap();

    // create buffer of all vertexes
    let mut dest_buffer_offset_elems = 0;
    for blas_vertex_buffer in blas_vertex_buffers {
        let mut cbi = CopyBufferInfoTyped::buffers(blas_vertex_buffer.clone(), dest_buffer.clone());
        cbi.regions[0].dst_offset = dest_buffer_offset_elems;
        builder.copy_buffer(cbi).unwrap();
        dest_buffer_offset_elems += blas_vertex_buffer.len();
    }

    // submit buffer
    let future = builder.build().unwrap().execute(queue).unwrap().boxed();

    (dest_buffer, future)
}

fn create_top_level_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    bottom_level_acceleration_structures: &[&AccelerationStructure],
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
    let mut instances = bottom_level_acceleration_structures
        .iter()
        .map(
            |&bottom_level_acceleration_structure| AccelerationStructureInstance {
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(
                    0,
                    GeometryInstanceFlags::TRIANGLE_FACING_CULL_DISABLE.into(),
                ),
                acceleration_structure_reference: bottom_level_acceleration_structure
                    .device_address()
                    .get(),
                ..Default::default()
            },
        )
        .collect::<Vec<_>>();

    // if there are no instances, then we create a dummy inactive instance to permit the TLAS to be built
    if instances.len() == 0 {
        instances.push(AccelerationStructureInstance {
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
            ..Default::default()
        });
    }

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
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE,
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
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::TopLevel,
        build_info,
        &[bottom_level_acceleration_structures.len() as u32],
        build_range_infos,
    )
}

fn create_bottom_level_acceleration_structure<T: BufferContents + vertex_input::Vertex>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    vertex_buffers: &[&Subbuffer<[T]>],
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
    let description = T::per_vertex();

    assert_eq!(description.stride, std::mem::size_of::<T>() as u32);

    let mut triangles = vec![];
    let mut max_primitive_counts = vec![];
    let mut build_range_infos = vec![];

    for &vertex_buffer in vertex_buffers {
        let primitive_count = vertex_buffer.len() as u32 / 3;
        triangles.push(AccelerationStructureGeometryTrianglesData {
            flags: GeometryFlags::OPAQUE,
            vertex_data: Some(vertex_buffer.clone().into_bytes()),
            vertex_stride: description.stride,
            max_vertex: vertex_buffer.len() as _,
            index_data: None,
            transform_data: None,
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
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE,
        mode: BuildAccelerationStructureMode::Build,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    build_acceleration_structure(
        memory_allocator,
        command_buffer_allocator,
        queue,
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
            usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE,
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
    Buffer::new_slice::<u8>(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        size,
    )
    .unwrap()
}

fn build_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    ty: AccelerationStructureType,
    mut build_info: AccelerationStructureBuildGeometryInfo,
    max_primitive_counts: &[u32],
    build_range_infos: impl IntoIterator<Item = AccelerationStructureBuildRangeInfo>,
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
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

    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    unsafe {
        builder
            .build_acceleration_structure(build_info, build_range_infos.into_iter().collect())
            .unwrap();
    }

    let command_buffer = builder.build().unwrap();

    let future = command_buffer.execute(queue).unwrap().boxed();

    (acceleration_structure, future)
}
