use core::panic;
use std::sync::Arc;

use image::RgbaImage;
use nalgebra::{Point3, Vector2, Vector3};
use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorBindingFlags,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Features, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{
        sampler::Sampler, view::ImageView, Image, ImageCreateInfo, ImageFormatInfo, ImageType,
        ImageUsage,
    },
    instance::Instance,
    memory::{
        self,
        allocator::{
            AllocationCreateInfo, MemoryAllocatePreference, MemoryTypeFilter,
            StandardMemoryAllocator,
        },
    },
    pipeline::{
        compute::ComputePipelineCreateInfo, graphics::viewport::Viewport,
        layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline,
        PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::window::Window;

use super::{
    bvh::BvhNode,
    pathtrace_shader,
    vertex::{InstanceData, Vertex3D},
};

pub fn get_device_for_rendering_on(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
    let device_extensions = DeviceExtensions {
        khr_acceleration_structure: true,
        khr_ray_query: true,
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let features = Features {
        acceleration_structure: true,
        buffer_device_address: true,
        dynamic_rendering: true,
        ray_query: true,
        shader_int8: true,
        shader_int64: true,
        storage_buffer8_bit_access: true,
        runtime_descriptor_array: true,
        descriptor_binding_variable_descriptor_count: true,
        ..Features::empty()
    };
    let (physical_device, general_queue_family_index, transfer_queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            // find a general purpose queue
            let general_queue_family_index = p
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags
                        .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                });

            // find a transfer-only queue (this will be fast for transfers)
            let transfer_queue_family_index = p
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // has transfer and sparse binding only
                    q.queue_flags == QueueFlags::TRANSFER | QueueFlags::SPARSE_BINDING
                });

            match (general_queue_family_index, transfer_queue_family_index) {
                (Some(q), Some(t)) => Some((p, q as u32, t as u32)),
                _ => None,
            }
        })
        .min_by_key(|(p, _, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no suitable physical device found");

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: features,
            queue_create_infos: vec![
                QueueCreateInfo {
                    queue_family_index: general_queue_family_index,
                    ..Default::default()
                },
                QueueCreateInfo {
                    queue_family_index: transfer_queue_family_index,
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    )
    .unwrap();

    let general_queue = queues.next().unwrap();
    let transfer_queue = queues.next().unwrap();

    (device, general_queue, transfer_queue)
}

fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface>,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    // Querying the capabilities of the surface. When we create the swapchain we can only
    // pass values that are allowed by the capabilities.
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

    // Please take a look at the docs for the meaning of the parameters we didn't mention.
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: 8,
            image_format: Format::B8G8R8A8_SRGB,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::TRANSFER_DST,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),

            ..Default::default()
        },
    )
    .unwrap()
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    memory_allocator: Arc<StandardMemoryAllocator>,
    images: &[Arc<Image>],
) -> Vec<Subbuffer<[u8]>> {
    let render_dests = images
        .iter()
        .map(|image| {
            let extent = image.extent();

            Buffer::new_slice::<u8>(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                    ..Default::default()
                },
                (extent[0] * extent[1] * 4) as u64,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    render_dests
}

pub fn get_surface_extent(surface: &Surface) -> [u32; 2] {
    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
    window.inner_size().into()
}

pub struct Renderer {
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    swapchain: Arc<Swapchain>,
    material_descriptor_set: Arc<PersistentDescriptorSet>,
    render_dests: Vec<Subbuffer<[u8]>>,
    swapchain_images: Vec<Arc<Image>>,
    pipeline: Arc<ComputePipeline>,
    wdd_needs_rebuild: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    frame_count: u32,
}

fn load_textures(
    textures: Vec<RgbaImage>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView>> {
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let mut image_views = vec![];

    for texture in textures {
        let extent = [texture.width(), texture.height(), 1];

        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            texture.into_raw(),
        )
        .unwrap();

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_SRGB,
                extent,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                upload_buffer,
                image.clone(),
            ))
            .unwrap();

        image_views.push(ImageView::new_default(image).unwrap());
    }

    let future = builder.build().unwrap().execute(queue.clone()).unwrap();

    future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    image_views
}

impl Renderer {
    pub fn new(
        surface: Arc<Surface>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        texture_atlas: Vec<(RgbaImage, RgbaImage, RgbaImage)>,
    ) -> Renderer {
        let texture_atlas = texture_atlas
            .into_iter()
            .flat_map(|(reflectivity, emissivity, metallicity)| {
                [reflectivity, emissivity, metallicity]
            })
            .collect::<Vec<_>>();

        let device = memory_allocator.device().clone();

        let (swapchain, swapchain_images) = create_swapchain(device.clone(), surface.clone());

        let pipeline = {
            let cs = pathtrace_shader::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(cs);

            let layout = {
                let mut layout_create_info =
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&[stage.clone()]);

                // Adjust the info for set 0, binding 1 to make it variable with texture_atlas.len() descriptors.
                let binding = layout_create_info.set_layouts[0]
                    .bindings
                    .get_mut(&1)
                    .unwrap();
                binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                binding.descriptor_count = texture_atlas.len() as u32;

                PipelineLayout::new(
                    device.clone(),
                    layout_create_info
                        .into_pipeline_layout_create_info(device.clone())
                        .unwrap(),
                )
                .unwrap()
            };

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let render_dests = window_size_dependent_setup(memory_allocator.clone(), &swapchain_images);

        let texture_atlas = load_textures(
            texture_atlas,
            queue.clone(),
            command_buffer_allocator.clone(),
            memory_allocator.clone(),
        );

        let sampler = Sampler::new(device.clone(), Default::default()).unwrap();

        let material_descriptor_set = PersistentDescriptorSet::new_variable(
            &descriptor_set_allocator,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            texture_atlas.len() as u32,
            [
                WriteDescriptorSet::sampler(0, sampler),
                WriteDescriptorSet::image_view_array(1, 0, texture_atlas),
            ],
            [],
        )
        .unwrap();

        Renderer {
            surface,
            command_buffer_allocator,
            previous_frame_end: Some(sync::now(device.clone()).boxed()),
            device,
            queue,
            swapchain,
            pipeline,
            descriptor_set_allocator,
            render_dests,
            swapchain_images,
            memory_allocator,
            wdd_needs_rebuild: false,
            material_descriptor_set,
            frame_count: 0,
        }
    }

    pub fn n_swapchain_images(&self) -> usize {
        self.swapchain_images.len()
    }

    pub fn rebuild(&mut self, extent: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = new_swapchain;
        self.render_dests = window_size_dependent_setup(self.memory_allocator.clone(), &new_images);
        self.swapchain_images = new_images;
    }

    pub fn render(
        &mut self,
        build_future: Box<dyn GpuFuture>,
        top_level_acceleration_structure: Arc<AccelerationStructure>,
        instance_data: Subbuffer<[InstanceData]>,
        luminance_bvh: Subbuffer<[BvhNode]>,
        eye: Point3<f32>,
        front: Vector3<f32>,
        right: Vector3<f32>,
        up: Vector3<f32>,
        samples: u32,
    ) {
        // Do not draw frame when screen dimensions are zero.
        // On Windows, this can occur from minimizing the application.
        let extent = get_surface_extent(&self.surface);
        if extent[0] == 0 || extent[1] == 0 {
            return;
        }
        // free memory
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        // Whenever the window resizes we need to recreate everything dependent on the window size.
        // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
        if self.wdd_needs_rebuild {
            self.rebuild(extent);
            self.wdd_needs_rebuild = false;
            println!("rebuilt swapchain");
        }

        // This operation returns the index of the image that we are allowed to draw upon.
        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    println!("swapchain out of date (at acquire)");
                    self.wdd_needs_rebuild = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            self.wdd_needs_rebuild = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let per_frame_descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.pipeline.layout().set_layouts().get(1).unwrap().clone(),
            [
                WriteDescriptorSet::acceleration_structure(0, top_level_acceleration_structure),
                WriteDescriptorSet::buffer(1, instance_data),
                WriteDescriptorSet::buffer(2, luminance_bvh),
                WriteDescriptorSet::buffer(3, self.render_dests[image_index as usize].clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                (
                    self.material_descriptor_set.clone(),
                    per_frame_descriptor_set,
                ),
            )
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                pathtrace_shader::Camera {
                    eye: eye.coords,
                    front,
                    right,
                    up,
                    screen_size: [extent[0], extent[1]].into(),
                    frame: self.frame_count,
                    samples,
                },
            )
            .unwrap()
            .dispatch([extent[0] / 32 + 1, extent[1] / 32 + 1, 1])
            .unwrap()
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                self.render_dests[image_index as usize].clone(),
                self.swapchain_images[image_index as usize].clone(),
            ))
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(build_future)
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.wdd_needs_rebuild = true;
                println!("swapchain out of date (at flush)");
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }

        self.frame_count = self.frame_count.wrapping_add(1);
    }
}
