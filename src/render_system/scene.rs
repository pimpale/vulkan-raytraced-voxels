use std::{collections::BTreeMap, fmt::Debug, sync::Arc};

use nalgebra::{Isometry3, Matrix4};
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

pub struct Object<Vertex> {
    object: Vec<Vertex>,
    isometry: Isometry3<f32>,
    vertex_buffer: Subbuffer<[Vertex]>,
    blas: Arc<AccelerationStructure>,
    blas_build_future: FenceSignalFuture<Box<dyn GpuFuture>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopLevelAccelerationStructureState {
    UpToDate,
    NeedsUpdate,
    NeedsRebuild,
}

/// Corresponds to a TLAS
pub struct Scene<K, Vertex> {
    general_queue: Arc<Queue>,
    transfer_queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: BTreeMap<K, Option<Object<Vertex>>>,
    tlas: Arc<AccelerationStructure>,
    instance_vertex_buffer_addresses: Subbuffer<[u64]>,
    tlas_state: TopLevelAccelerationStructureState,
}

#[allow(dead_code)]
impl<K, Vertex> Scene<K, Vertex>
where
    Vertex: vertex_input::Vertex + Default + Clone + BufferContents + Debug,
    K: Ord + Clone + std::cmp::Eq + std::hash::Hash,
{
    pub fn new(
        general_queue: Arc<Queue>,
        transfer_queue: Arc<Queue>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> Scene<K, Vertex> {
        // assert that the vertex type must have a field called position
        assert!(Vertex::per_vertex().members.contains_key("position"));

        // in order to build the tlas, we need to have some data. We initialize it with a degenerate single triangle to permit this
        let degenerate_triangle = vec![Vertex::default(); 3];

        let blas_vertex_buffer =
            blas_vertex_buffer(memory_allocator.clone(), [&degenerate_triangle]);

        let (blas, blas_build_future) = create_bottom_level_acceleration_structure(
            None,
            memory_allocator.clone(),
            &command_buffer_allocator,
            general_queue.clone(),
            &[&blas_vertex_buffer],
            Isometry3::identity(),
        );

        // await build
        blas_build_future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let (tlas, tlas_build_future) = create_top_level_acceleration_structure(
            None,
            memory_allocator.clone(),
            &command_buffer_allocator,
            general_queue.clone(),
            &[&blas],
        );

        // await build
        tlas_build_future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        // create instance vertex buffer addresses
        let instance_vertex_buffer_addresses = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![blas_vertex_buffer.device_address().unwrap().get()],
        )
        .unwrap();

        Scene {
            general_queue,
            transfer_queue,
            command_buffer_allocator,
            memory_allocator,
            objects: BTreeMap::new(),
            tlas,
            instance_vertex_buffer_addresses,
            tlas_state: TopLevelAccelerationStructureState::UpToDate,
        }
    }

    // adds a new object to the scene with the given isometry
    pub fn add_object(&mut self, key: K, object: Vec<Vertex>, isometry: Isometry3<f32>) {
        if object.len() == 0 {
            self.objects.insert(key, None);
            return;
        }

        let vertex_buffer = blas_vertex_buffer(self.memory_allocator.clone(), [&object]);
        let (blas, blas_build_future) = create_bottom_level_acceleration_structure(
            None,
            self.memory_allocator.clone(),
            &self.command_buffer_allocator,
            self.general_queue.clone(),
            &[&vertex_buffer],
            isometry,
        );
        self.objects.insert(
            key,
            Some(Object {
                object,
                isometry,
                vertex_buffer,
                blas,
                blas_build_future: blas_build_future.then_signal_fence_and_flush().unwrap(),
            }),
        );
        self.tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
    }

    // updates the isometry of the object with the given key
    pub fn update_object(&mut self, key: K, isometry: Isometry3<f32>) {
        match self.objects.get_mut(&key) {
            Some(Some(object)) => {
                object.isometry = isometry;
                object.blas_build_future.wait(None).unwrap();
                let (blas, blas_build_future) = create_bottom_level_acceleration_structure(
                    Some(object.blas.clone()),
                    self.memory_allocator.clone(),
                    &self.command_buffer_allocator,
                    self.general_queue.clone(),
                    &[&object.vertex_buffer],
                    isometry,
                );
                object.blas = blas;
                object.blas_build_future = blas_build_future.then_signal_fence_and_flush().unwrap();
                if self.tlas_state == TopLevelAccelerationStructureState::UpToDate {
                    self.tlas_state = TopLevelAccelerationStructureState::NeedsUpdate;
                }
            }
            Some(None) => {}
            None => panic!("object with key does not exist"),
        }
    }

    pub fn remove_object(&mut self, key: K) {
        let removed = self.objects.remove(&key);
        if removed.is_some() {
            self.tlas_state = TopLevelAccelerationStructureState::NeedsRebuild;
        }
    }

    pub fn tlas(&mut self) -> (Arc<AccelerationStructure>, Subbuffer<[u64]>) {
        // need to update instance vertex buffer addresses if an object was added or removed
        if self.tlas_state == TopLevelAccelerationStructureState::NeedsRebuild {
            let instance_vertex_buffer_addresses = self
                .objects
                .values()
                .flatten()
                .map(|object| object.vertex_buffer.device_address().unwrap().get())
                .collect::<Vec<_>>();

            self.instance_vertex_buffer_addresses = Buffer::from_iter(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                instance_vertex_buffer_addresses,
            )
            .unwrap();
        }

        if self.tlas_state != TopLevelAccelerationStructureState::UpToDate {
            // await all the blas futures on the cpu
            for object in self.objects.values().flatten() {
                object.blas_build_future.wait(None).unwrap();
            }

            let previous_acceleration_structure = match self.tlas_state {
                TopLevelAccelerationStructureState::NeedsUpdate => Some(self.tlas.clone()),
                TopLevelAccelerationStructureState::NeedsRebuild => None,
                _ => unreachable!(),
            };

            // initialize tlas build
            let (tlas, tlas_build_future) = create_top_level_acceleration_structure(
                previous_acceleration_structure,
                self.memory_allocator.clone(),
                &self.command_buffer_allocator,
                self.general_queue.clone(),
                &self
                    .objects
                    .values()
                    .flatten()
                    .map(|Object { blas, .. }| blas as &AccelerationStructure)
                    .collect::<Vec<_>>(),
            );

            // actually submit acceleration structure build future
            let tlas_build_future = tlas_build_future.then_signal_fence_and_flush().unwrap();

            tlas_build_future.wait(None).unwrap();

            // update state
            self.tlas = tlas;
        }

        // at this point the tlas is up to date
        self.tlas_state = TopLevelAccelerationStructureState::UpToDate;

        // return the tlas
        return (
            self.tlas.clone(),
            self.instance_vertex_buffer_addresses.clone(),
        );
    }
}

fn blas_vertex_buffer<'a, Vertex, Container>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: Container,
) -> Subbuffer<[Vertex]>
where
    Container: IntoIterator<Item = &'a Vec<Vertex>>,
    Vertex: Default + Clone + BufferContents,
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
    .unwrap()
}

fn create_top_level_acceleration_structure(
    previous_acceleration_structure: Option<Arc<AccelerationStructure>>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    bottom_level_acceleration_structures: &[&AccelerationStructure],
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
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
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE,
        mode: match previous_acceleration_structure {
            Some(p) => BuildAccelerationStructureMode::Update(p),
            None => BuildAccelerationStructureMode::Build,
        },
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
    previous_acceleration_structure: Option<Arc<AccelerationStructure>>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    vertex_buffers: &[&Subbuffer<[T]>],
    isometry: Isometry3<f32>,
) -> (Arc<AccelerationStructure>, Box<dyn GpuFuture>) {
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
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE
            | BuildAccelerationStructureFlags::ALLOW_UPDATE,
        mode: match previous_acceleration_structure {
            Some(p) => BuildAccelerationStructureMode::Update(p),
            None => BuildAccelerationStructureMode::Build,
        },
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
