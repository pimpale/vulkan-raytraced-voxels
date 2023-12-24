use std::{collections::HashMap, sync::Arc};

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter}, command_buffer::{allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage}, device::Queue, acceleration_structure::{AccelerationStructure, AccelerationStructureInstance, GeometryInstanceFlags, AccelerationStructureGeometries, AccelerationStructureGeometryInstancesData, GeometryFlags, AccelerationStructureGeometryInstancesDataType, AccelerationStructureBuildGeometryInfo, BuildAccelerationStructureMode, AccelerationStructureBuildRangeInfo, AccelerationStructureType, AccelerationStructureGeometryTrianglesData, BuildAccelerationStructureFlags, AccelerationStructureCreateInfo, AccelerationStructureBuildSizesInfo, AccelerationStructureBuildType}, Packed24_8, pipeline::graphics::vertex_input::Vertex,
};

pub struct Scene<K, Vertex> {
    objects: HashMap<K, Vec<Vertex>>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    vertex_buffer: Option<Subbuffer<[Vertex]>>,
    vertex_buffer_needs_update: bool,
}

#[allow(dead_code)]
impl<K, Vertex> Scene<K, Vertex>
where
    Vertex: Clone + BufferContents,
    K: std::cmp::Eq + std::hash::Hash,
{
    pub fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        objects: HashMap<K, Vec<Vertex>>,
    ) -> Scene<K, Vertex> {
        Scene {
            vertex_buffer: vertex_buffer(memory_allocator.clone(), objects.values()),
            objects,
            memory_allocator,
            vertex_buffer_needs_update: false,
        }
    }

    pub fn add_object(&mut self, key: K, object: Vec<Vertex>) {
        self.objects.insert(key, object);
        self.vertex_buffer_needs_update = true;
    }

    pub fn remove_object(&mut self, key: K) {
        let removed = self.objects.remove(&key);
        if removed.is_some() {
            self.vertex_buffer_needs_update = true;
        }
    }

    pub fn objects(&self) -> &HashMap<K, Vec<Vertex>> {
        &self.objects
    }

    pub fn vertex_buffer(&mut self) -> Option<Subbuffer<[Vertex]>> {
        if self.vertex_buffer_needs_update {
            self.vertex_buffer =
                vertex_buffer(self.memory_allocator.clone(), self.objects.values());
            self.vertex_buffer_needs_update = false;
        }
        return self.vertex_buffer.clone();
    }
}

fn vertex_buffer<'a, Vertex, Container>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    objects: Container,
) -> Option<Subbuffer<[Vertex]>>
where
    Container: IntoIterator<Item = &'a Vec<Vertex>>,
    Vertex: Clone + BufferContents,
{
    let vertexes = objects
        .into_iter()
        .flat_map(|o| o.iter())
        .cloned()
        .collect::<Vec<Vertex>>();
    if vertexes.len() == 0 {
        return None;
    } else {
        let buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
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

        return Some(buffer);
    }
}


fn create_top_level_acceleration_structure(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    bottom_level_acceleration_structures: &[&AccelerationStructure],
) -> Arc<AccelerationStructure> {
    let instances = bottom_level_acceleration_structures
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
        memory_allocator,
        command_buffer_allocator,
        queue,
        AccelerationStructureType::TopLevel,
        build_info,
        &[bottom_level_acceleration_structures.len() as u32],
        build_range_infos,
    )
}

fn create_bottom_level_acceleration_structure<T: BufferContents + Vertex>(
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: Arc<Queue>,
    vertex_buffers: &[&Subbuffer<[T]>],
) -> Arc<AccelerationStructure> {
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
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
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
    command_buffer
        .execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration_structure
}
