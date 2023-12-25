use std::sync::Arc;

use nalgebra::{Matrix4, Point3, Vector3};
use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Features, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::Instance,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexBufferDescription, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentStoreOp, Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::{spirv::ExecutionModel, EntryPoint},
    swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::window::Window;

use super::shader::{fs, vs};

pub fn get_device_for_rendering_on(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
) -> (Arc<Device>, Arc<Queue>) {
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
        ..Features::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
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
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    (device, queue)
}

#[derive(Clone, BufferContents, Vertex)]
#[repr(C)]
struct Vertex2D {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

// The quad buffer that covers the entire surface
static QUAD: [Vertex2D; 6] = [
    Vertex2D {
        position: [-1.0, -1.0],
    },
    Vertex2D {
        position: [-1.0, 1.0],
    },
    Vertex2D {
        position: [1.0, -1.0],
    },
    Vertex2D {
        position: [1.0, 1.0],
    },
    Vertex2D {
        position: [1.0, -1.0],
    },
    Vertex2D {
        position: [-1.0, 1.0],
    },
];

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

    // Choosing the internal format that the images will have.
    let image_format = device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

    // Please take a look at the docs for the meaning of the parameters we didn't mention.
    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,
            image_format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
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
fn window_size_dependent_setup(images: &[Arc<Image>]) -> (Vec<Arc<ImageView>>, Viewport) {
    let extent = images[0].extent();

    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [extent[0] as f32, extent[1] as f32],
        depth_range: 0.0..=1.0,
    };

    let image_views = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect::<Vec<_>>();

    (image_views, viewport)
}

pub fn get_surface_extent(surface: &Surface) -> [u32; 2] {
    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
    window.inner_size().into()
}

pub struct Renderer {
    viewport: Viewport,
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    quad_buffer: Subbuffer<[Vertex2D]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    pipeline: Arc<GraphicsPipeline>,
    wdd_needs_rebuild: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl Renderer {
    pub fn new(
        surface: Arc<Surface>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Renderer {
        let device = memory_allocator.device().clone();

        let (swapchain, images) = create_swapchain(device.clone(), surface.clone());

        let pipeline = {
            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = Vertex2D::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(swapchain.image_format())],
                ..Default::default()
            };
            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.color_attachment_formats.len() as u32,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let (attachment_image_views, viewport) = window_size_dependent_setup(&images);

        let quad_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            QUAD.iter().cloned(),
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
            attachment_image_views,
            viewport,
            memory_allocator,
            wdd_needs_rebuild: false,
            quad_buffer,
        }
    }

    pub fn rebuild(&mut self, extent: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

        let (new_attachment_image_views, new_viewport) = window_size_dependent_setup(&new_images);

        self.swapchain = new_swapchain;
        self.attachment_image_views = new_attachment_image_views;
        self.viewport = new_viewport;
    }

    pub fn render(
        &mut self,
        top_level_acceleration_structure: Arc<AccelerationStructure>,
        eye: Point3<f32>,
        front: Vector3<f32>,
        right: Vector3<f32>,
        up: Vector3<f32>,
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

        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::acceleration_structure(
                0,
                top_level_acceleration_structure,
            )],
            [],
        )
        .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    store_op: AttachmentStoreOp::Store,
                    ..RenderingAttachmentInfo::image_view(
                        self.attachment_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.quad_buffer.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                fs::Camera {
                    eye: <[f32; 3]>::from(eye).into(),
                    front: <[f32; 3]>::from(front).into(),
                    right: <[f32; 3]>::from(right).into(),
                    up: <[f32; 3]>::from(up).into(),
                    aspect: extent[0] as f32 / extent[1] as f32,
                },
            )
            .unwrap()
            .draw(self.quad_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_rendering()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
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
    }
}
