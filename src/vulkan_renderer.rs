use crate::config;
use crate::utils;
use ash::ext::debug_utils;
use ash::{vk, Entry};
use core::panic;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::error::Error;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use winit::raw_window_handle::HasDisplayHandle;
use winit::raw_window_handle::HasWindowHandle;
use winit::window::Window;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        _ => "[Unknown]",
    };
    let message = CStr::from_ptr((*p_callback_data).p_message);
    println!("[VK_Debug]{}{}{:?}", severity, types, message);
    vk::FALSE
}

pub struct VulkanRenderer {
    entry: Entry, //we aren't allowed to call any Vulkan functions after entry is dropped!
    instance: ash::Instance,
    debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface: vk::SurfaceKHR,
    surface_instance: ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    logical_device: ash::Device,
    graphics_queue: vk::Queue,
    presentation_queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl VulkanRenderer {
    pub fn new(window: &Window) -> Result<VulkanRenderer, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        let (instance, validation_layer_enabled) = VulkanRenderer::create_instance(window, &entry)?;
        let debug_utils_messenger = if validation_layer_enabled {
            Some(VulkanRenderer::setup_debug_messenger(&entry, &instance)?)
        } else {
            None
        };
        let (surface, surface_instance) =
            VulkanRenderer::create_surface(&entry, &instance, window)?;
        let physical_device = Self::get_physical_device(&instance, &surface_instance, surface);
        let (logical_device, graphics_queue, presentation_queue) =
            VulkanRenderer::create_logical_device(
                &instance,
                physical_device,
                vk::PhysicalDeviceFeatures::default(),
                &surface_instance,
                surface,
            )?;
        let (
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = Self::create_swap_chain(
            &instance,
            &surface_instance,
            surface,
            physical_device,
            &logical_device,
        );

        let swapchain_image_views =
            Self::create_image_views(&logical_device, swapchain_image_format, &swapchain_images);

        let render_pass = Self::create_render_pass(&logical_device, swapchain_image_format);

        let (pipeline_layout, graphics_pipeline) =
            Self::create_graphics_pipeline(&logical_device, render_pass);

        let swapchain_framebuffers = Self::create_framebuffers(
            &logical_device,
            &swapchain_image_views,
            render_pass,
            swapchain_extent,
        );

        let command_pool = Self::create_command_pool(
            &instance,
            &logical_device,
            physical_device,
            surface,
            &surface_instance,
        );

        let command_buffer = Self::create_command_buffer(&logical_device, command_pool);

        Ok(VulkanRenderer {
            entry,
            instance,
            debug_utils_messenger,
            surface,
            surface_instance,
            physical_device,
            logical_device,
            graphics_queue,
            presentation_queue,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            graphics_pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffer,
        })
    }

    fn create_instance(
        window: &Window,
        entry: &Entry,
    ) -> Result<(ash::Instance, bool), Box<dyn Error>> {
        let (required_layers, validation_layer_enabled) = if cfg!(debug_assertions) {
            let available_layers = VulkanRenderer::get_available_instance_layers(entry);
            let required_layers = VulkanRenderer::get_required_instance_layers();
            let layer_enabled =
                VulkanRenderer::check_validation_layer_support(&available_layers, &required_layers);
            if layer_enabled {
                println!("Enabling Vulkan Validation Layers");
            } else {
                println!("Can't Enable Validation Layers since required layers are missing.");
            }
            (required_layers, layer_enabled)
        } else {
            println!("Detected release build. Disabling Vulkan validation layers.");
            (Vec::new(), false)
        };
        let required_layers_raw = utils::vec_cstring_to_vec_pointers(&required_layers);

        let required_extensions =
            VulkanRenderer::get_required_extensions(window, validation_layer_enabled)?;

        let app_name =
            CString::new(config::WINDOW_NAME).expect("Hardcoded constant should never fail");
        let engine_name =
            CString::new(config::ENGINE_NAME).expect("Hardcoded constant should never fail");
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: config::APPLICATION_VERSION,
            p_engine_name: engine_name.as_ptr(),
            engine_version: config::ENGINE_VERSION,
            api_version: config::VK_API_VERSION,
            ..Default::default()
        };

        let debug_create_info = VulkanRenderer::create_debug_utils_messenger_create_info();

        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_extension_count: required_extensions.len() as u32,
            pp_enabled_extension_names: required_extensions.as_ptr(),
            p_next: if validation_layer_enabled {
                (&debug_create_info) as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
            } else {
                ptr::null()
            },
            flags: vk::InstanceCreateFlags::empty(),
            enabled_layer_count: required_layers.len() as u32,
            pp_enabled_layer_names: required_layers_raw.as_ptr(),
            ..Default::default()
        };

        // safety: needs entry to be valid, which is ensured by reference
        let instance = unsafe { entry.create_instance(&create_info, Option::None)? };
        Ok((instance, validation_layer_enabled))
    }

    fn get_available_instance_layers(entry: &Entry) -> Vec<CString> {
        let layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("Could not enumerate instance layer properties")
        };
        let instance_layers: Vec<CString> = layer_properties
            .iter()
            .map(|prop| {
                CString::from(
                    prop.layer_name_as_c_str()
                        .expect("Could not convert layer name to UTF C Str"),
                )
            })
            .collect();

        println!("Available Instance Layers: ");
        println!("==================");
        for layer in instance_layers.iter() {
            println!("{:?}", layer);
        }
        println!("==================");
        println!();

        instance_layers
    }

    fn get_required_instance_layers() -> Vec<CString> {
        let required_layers: Vec<CString> = config::VALIDATION_LAYERS
            .iter()
            .map(|layer| {
                CString::new(*layer).expect("Hardcoded constant should never fail in conversion")
            })
            .collect();

        println!("Required Instance Layers: ");
        println!("==================");
        for layer in required_layers.iter() {
            println!("{:?}", layer);
        }
        println!("==================");
        println!();

        required_layers
    }

    fn check_validation_layer_support(
        available_layers: &[CString],
        required_layers: &[CString],
    ) -> bool {
        for required_layer in required_layers.iter() {
            let mut layer_found = false;
            for available_layer in available_layers.iter() {
                if required_layer == available_layer {
                    layer_found = true;
                    break;
                }
            }
            if !layer_found {
                println!(
                    "Required layer {:?} not found! Disabling validation layer!",
                    required_layer
                );
                return false;
            }
        }
        true
    }

    fn get_required_extensions(
        window: &Window,
        validation_layer_enabled: bool,
    ) -> Result<Vec<*const c_char>, Box<dyn Error>> {
        let mut required_extensions =
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?.to_vec();
        if validation_layer_enabled {
            required_extensions.push(debug_utils::NAME.as_ptr());
        }
        Ok(required_extensions)
    }

    fn create_debug_utils_messenger_create_info<'a>() -> vk::DebugUtilsMessengerCreateInfoEXT<'a> {
        vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            p_next: ptr::null(),
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(vulkan_debug_callback),
            p_user_data: ptr::null_mut(),
            ..Default::default()
        }
    }

    fn setup_debug_messenger(
        entry: &Entry,
        instance: &ash::Instance,
    ) -> Result<vk::DebugUtilsMessengerEXT, Box<dyn Error>> {
        let create_info = VulkanRenderer::create_debug_utils_messenger_create_info();
        let debug_utils_instance = debug_utils::Instance::new(entry, instance);
        let debug_utils_messenger =
            unsafe { debug_utils_instance.create_debug_utils_messenger(&create_info, None) }?;
        Ok(debug_utils_messenger)
    }

    fn get_physical_device(
        instance: &ash::Instance,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> vk::PhysicalDevice {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("There should be atleast one working Vulkan device!")
        };

        println!(
            "Found {} devices with Vulkan support",
            physical_devices.len()
        );

        let mut suitable_physical_devices: Vec<vk::PhysicalDevice> = physical_devices
            .into_iter()
            .filter(|device| Self::is_device_suitable(instance, *device, surface_instance, surface))
            .collect();
        println!("Found {} suitable devices", suitable_physical_devices.len());

        suitable_physical_devices
            .sort_by_key(|device| Reverse(Self::get_device_suitability_score(instance, *device)));

        if suitable_physical_devices.is_empty() {
            panic!("No suitable devices found!")
        }

        let chosen_device = suitable_physical_devices[0];

        let device_properties = unsafe { instance.get_physical_device_properties(chosen_device) };
        let device_name = device_properties
            .device_name_as_c_str()
            .expect("Should be able to convert dev name to c_str");
        println!("Choosing device {:?}", device_name);
        chosen_device
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> bool {
        let queue_families_supported =
            Self::find_queue_families(instance, device, surface_instance, surface).is_complete();
        let extensions_supported = Self::check_device_extension_support(instance, device);
        let mut swapchain_adequate = false;
        if extensions_supported {
            let swap_chain_support =
                SwapChainSupportDetails::query_support_details(surface_instance, surface, device);
            swapchain_adequate = !swap_chain_support.surface_formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }
        queue_families_supported && extensions_supported && swapchain_adequate
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> bool {
        let supported_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .expect("Could not enumerate device extension properties")
        };
        let required_extensions = config::REQUIRED_DEVICE_EXTENSIONS;
        let cross_section = supported_extensions.iter().filter(|extension_prop| {
            required_extensions.contains(
                &extension_prop
                    .extension_name_as_c_str()
                    .expect("Could not convert extension name to  str")
                    .to_str()
                    .expect("Could not convert extension name to  str"),
            )
        });
        cross_section.count() == required_extensions.len()
    }

    fn find_queue_families(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> QueueFamilyIndices {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        let mut queue_family_indices = QueueFamilyIndices::new();
        for (idx, queue_family_property) in queue_family_properties.iter().enumerate() {
            if queue_family_property
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(idx as u32);
            }
            if unsafe {
                surface_instance
                    .get_physical_device_surface_support(device, idx as u32, surface)
                    .expect("Host does not have enough resources or smth")
            } {
                queue_family_indices.presentation_family = Some(idx as u32);
            }
        }
        queue_family_indices
    }

    fn get_device_suitability_score(instance: &ash::Instance, device: vk::PhysicalDevice) -> u64 {
        let device_properties = unsafe { instance.get_physical_device_properties(device) };
        let mut score = 0;
        score += match device_properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::CPU => 10,
            _ => 0,
        };
        score
    }

    fn create_logical_device(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        required_device_features: vk::PhysicalDeviceFeatures,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(ash::Device, vk::Queue, vk::Queue), Box<dyn Error>> {
        let queue_family_indices =
            Self::find_queue_families(instance, device, surface_instance, surface);
        let graphics_q_fam_idx = queue_family_indices.graphics_family.expect(
            "Graphics Family not found! Should not be possible since we checked for suitability!",
        );
        let present_q_fam_idx = queue_family_indices.presentation_family.expect(
            "Graphics Family not found! Should not be possible since we checked for suitability!",
        );
        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(graphics_q_fam_idx);
        unique_queue_families.insert(present_q_fam_idx);
        println!("Using Queue Families: {:?}", unique_queue_families);
        let mut queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::new();
        for queue_family_index in unique_queue_families {
            let device_queue_create_info = vk::DeviceQueueCreateInfo {
                s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
                p_next: ptr::null(),
                queue_family_index,
                queue_count: 1,
                p_queue_priorities: [1.0].as_ptr(),
                flags: vk::DeviceQueueCreateFlags::empty(),
                ..Default::default()
            };
            queue_create_infos.push(device_queue_create_info);
        }

        let required_extensions = config::REQUIRED_DEVICE_EXTENSIONS;
        let (_required_extension_names_cstr, required_extension_names_raw) =
            utils::vec_str_ref_to_vec_pointers(&required_extensions);
        let device_create_info = vk::DeviceCreateInfo {
            s_type: vk::StructureType::DEVICE_CREATE_INFO,
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_enabled_features: &required_device_features,
            p_next: ptr::null(),
            enabled_extension_count: required_extension_names_raw.len() as u32,
            pp_enabled_extension_names: required_extension_names_raw.as_ptr(),
            flags: vk::DeviceCreateFlags::empty(),
            ..Default::default()
        };

        let logical_device = unsafe { instance.create_device(device, &device_create_info, None)? };
        let graphics_queue = unsafe { logical_device.get_device_queue(graphics_q_fam_idx, 0) };
        let presentation_queue = unsafe { logical_device.get_device_queue(present_q_fam_idx, 0) };
        Ok((logical_device, graphics_queue, presentation_queue))
    }

    fn create_surface(
        entry: &Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> Result<(vk::SurfaceKHR, ash::khr::surface::Instance), Box<dyn Error>> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle().expect("Should work").as_raw(),
                window.window_handle().expect("Should work").as_raw(),
                None,
            )?
        };
        let surface_instance = ash::khr::surface::Instance::new(entry, instance);
        Ok((surface, surface_instance))
    }

    fn create_swap_chain(
        instance: &ash::Instance,
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
        logical_device: &ash::Device,
    ) -> (
        ash::khr::swapchain::Device,
        vk::SwapchainKHR,
        Vec<vk::Image>,
        vk::Format,
        vk::Extent2D,
    ) {
        let support_details =
            SwapChainSupportDetails::query_support_details(surface_instance, surface, device);

        let surface_format =
            VulkanRenderer::choose_swap_surface_format(&support_details.surface_formats);
        let present_mode = VulkanRenderer::choose_swap_present_mode(&support_details.present_modes);
        let extent = VulkanRenderer::choose_swap_extent(&support_details.capabilities);

        let mut image_count = support_details.capabilities.min_image_count + 1;
        if support_details.capabilities.max_image_count > 0 {
            image_count = image_count.min(support_details.capabilities.max_image_count);
        }

        let queue_family_indices =
            Self::find_queue_families(instance, device, surface_instance, surface);
        if !queue_family_indices.is_complete() {
            panic!("Could not find suitable queue families for swap chain creation");
        }
        let indices_array = [
            queue_family_indices
                .graphics_family
                .expect("Should be filled since we checked for it"),
            queue_family_indices
                .presentation_family
                .expect("Should be filled since we checked for it"),
        ];
        let (image_sharing_mode, queue_fam_index_count, p_queue_fam_indices) =
            if queue_family_indices.graphics_family != queue_family_indices.presentation_family {
                (vk::SharingMode::CONCURRENT, 2, indices_array.as_ptr())
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, ptr::null())
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count: queue_fam_index_count,
            p_queue_family_indices: p_queue_fam_indices,
            pre_transform: support_details.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            ..Default::default()
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(instance, logical_device);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("Could not create swapchain")
        };
        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Could not get swapchain images")
        };

        (
            swapchain_loader,
            swapchain,
            swapchain_images,
            surface_format.format,
            extent,
        )
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        let desired_format = available_formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        });
        match desired_format {
            Some(format) => *format,
            None => *available_formats.first().expect("Should not be empty, since we already got so far in swap chain process, so we expect some format to be available"),
        }
    }

    fn choose_swap_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        let desired_mode = available_present_modes
            .iter()
            .find(|mode| **mode == vk::PresentModeKHR::MAILBOX);
        match desired_mode {
            Some(mode) => *mode,
            // FIFO is guaranteed to be available
            None => vk::PresentModeKHR::FIFO,
        }
    }

    fn choose_swap_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: utils::clamp(
                    config::WINDOW_WIDTH as u32,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: utils::clamp(
                    config::WINDOW_HEIGHT as u32,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }

    pub fn draw(&self) {}
    pub fn swap_buffers(&self) {}

    fn create_image_views(
        device: &ash::Device,
        format: vk::Format,
        swapchain_images: &[vk::Image],
    ) -> Vec<vk::ImageView> {
        let mut swapchain_views: Vec<vk::ImageView> = Vec::with_capacity(swapchain_images.len());
        for image in swapchain_images.iter() {
            let create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                p_next: ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                ..Default::default()
            };
            let image_view = unsafe {
                device
                    .create_image_view(&create_info, None)
                    .expect("Could not create image view")
            };
            swapchain_views.push(image_view);
        }
        swapchain_views
    }

    fn create_graphics_pipeline(
        logical_device: &ash::Device,
        render_pass: vk::RenderPass,
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let vert_shader_code = utils::read_shader_file("shaders_compiled/triangle_vert.spv");
        let frag_shader_code = utils::read_shader_file("shaders_compiled/triangle_frag.spv");
        println!("Vertex Shader Code len: {:?} bytes", vert_shader_code.len());
        println!(
            "Fragment Shader Code len: {:?} bytes",
            frag_shader_code.len()
        );
        let vert_shader_module =
            VulkanRenderer::create_shader_module(logical_device, &vert_shader_code);
        let frag_shader_module =
            VulkanRenderer::create_shader_module(logical_device, &frag_shader_code);

        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            ..Default::default()
        };

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            ..Default::default()
        };

        let shader_stage_infos = [vert_shader_stage_info, frag_shader_stage_info];

        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            dynamic_state_count: config::DYNAMIC_STATE.len() as u32,
            p_dynamic_states: config::DYNAMIC_STATE.as_ptr(),
            p_next: ptr::null(),
            flags: vk::PipelineDynamicStateCreateFlags::empty(),
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            viewport_count: 1,
            p_viewports: ptr::null(),
            scissor_count: 1,
            p_scissors: ptr::null(),
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            ..Default::default()
        };

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: ptr::null(),
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            ..Default::default()
        };

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            p_next: ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            ..Default::default()
        };

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            ..Default::default()
        };

        let multisample_info = vk::PipelineMultisampleStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            min_sample_shading: 1.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            blend_enable: vk::FALSE,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
        };

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: 0,
            p_set_layouts: ptr::null(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            logical_device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Could not create pipeline layout")
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            stage_count: shader_stage_infos.len() as u32,
            p_stages: shader_stage_infos.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly_info,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer_info,
            p_multisample_state: &multisample_info,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state_info,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            logical_device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("Could not create graphics pipeline")
                .first()
                .expect("Could not get first graphics pipeline")
                .to_owned()
        };

        unsafe {
            logical_device.destroy_shader_module(vert_shader_module, None);
            logical_device.destroy_shader_module(frag_shader_module, None);
        }
        (pipeline_layout, graphics_pipeline)
    }

    fn create_shader_module(logical_device: &ash::Device, byte_code: &[u8]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo {
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            code_size: byte_code.len(),
            p_code: byte_code.as_ptr() as *const u32,
            flags: vk::ShaderModuleCreateFlags::empty(),
            p_next: ptr::null(),
            ..Default::default()
        };
        unsafe {
            logical_device
                .create_shader_module(&create_info, None)
                .expect("Could not create shader module")
        }
    }

    fn create_render_pass(
        logical_device: &ash::Device,
        swapchain_image_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription {
            format: swapchain_image_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_depth_stencil_attachment: ptr::null(),
            flags: vk::SubpassDescriptionFlags::empty(),
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
            p_resolve_attachments: ptr::null(),
            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            dependency_count: 0,
            p_dependencies: ptr::null(),
            ..Default::default()
        };

        let render_pass = unsafe {
            logical_device
                .create_render_pass(&render_pass_info, None)
                .expect("Could not create render pass")
        };
        render_pass
    }

    fn create_framebuffers(
        logical_device: &ash::Device,
        swapchain_image_views: &[vk::ImageView],
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        let mut swapchain_frame_buffers = Vec::with_capacity(swapchain_image_views.len());

        for image_view in swapchain_image_views.iter() {
            let attachments = [*image_view];
            let framebuffer_info = vk::FramebufferCreateInfo {
                s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width: swapchain_extent.width,
                height: swapchain_extent.height,
                layers: 1,
                p_next: ptr::null(),
                flags: vk::FramebufferCreateFlags::empty(),
                ..Default::default()
            };
            let framebuffer = unsafe {
                logical_device
                    .create_framebuffer(&framebuffer_info, None)
                    .expect("Could not create framebuffer")
            };
            swapchain_frame_buffers.push(framebuffer);
        }
        swapchain_frame_buffers
    }

    fn create_command_pool(
        instance: &ash::Instance,
        logical_device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_instance: &ash::khr::surface::Instance,
    ) -> vk::CommandPool {
        let queue_family_indices =
            Self::find_queue_families(instance, physical_device, surface_instance, surface);
        let pool_info = vk::CommandPoolCreateInfo {
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            queue_family_index: queue_family_indices.graphics_family.unwrap(),
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            p_next: ptr::null(),
            ..Default::default()
        };
        unsafe {
            logical_device
                .create_command_pool(&pool_info, None)
                .expect("Could not create command pool")
        }
    }

    fn create_command_buffer(
        logical_device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            p_next: ptr::null(),
            ..Default::default()
        };
        let command_buffer = unsafe {
            logical_device
                .allocate_command_buffers(&alloc_info)
                .expect("Could not allocate command buffers")
        };
        command_buffer
            .first()
            .expect("Could not get first command buffer")
            .to_owned()
    }

    fn record_command_buffer(
        logical_device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        swapchain_framebuffers: Vec<vk::Framebuffer>,
        image_index: usize,
        extent: vk::Extent2D,
        graphics_pipeline: vk::Pipeline,
    ) {
        let begin_info = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: vk::CommandBufferUsageFlags::empty(),
            p_inheritance_info: ptr::null(),
            ..Default::default()
        };
        unsafe {
            logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Could not begin command buffer");
        }
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let renderpass_begin_info = vk::RenderPassBeginInfo {
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            render_pass,
            framebuffer: swapchain_framebuffers[image_index],
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            },
            clear_value_count: 1,
            p_clear_values: &clear_color,
            p_next: ptr::null(),
            ..Default::default()
        };
        unsafe {
            logical_device.cmd_begin_render_pass(
                command_buffer,
                &renderpass_begin_info,
                vk::SubpassContents::INLINE,
            );
            logical_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphics_pipeline,
            )
        }
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        unsafe {
            logical_device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            logical_device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            logical_device.cmd_draw(command_buffer, 3, 1, 0, 0);
            logical_device.cmd_end_render_pass(command_buffer);
            logical_device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
    }
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    presentation_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics_family: None,
            presentation_family: None,
        }
    }
    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.presentation_family.is_some()
    }
}

#[derive(Debug)]
struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    fn query_support_details(
        surface_instance: &ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
        device: vk::PhysicalDevice,
    ) -> SwapChainSupportDetails {
        let capabilities = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(device, surface)
                .expect("Could not get surface capabilities")
        };
        let surface_formats = unsafe {
            surface_instance
                .get_physical_device_surface_formats(device, surface)
                .expect("Could not get surface formats")
        };
        let present_modes = unsafe {
            surface_instance
                .get_physical_device_surface_present_modes(device, surface)
                .expect("Could not get present modes")
        };
        SwapChainSupportDetails {
            capabilities,
            surface_formats,
            present_modes,
        }
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            self.logical_device
                .destroy_command_pool(self.command_pool, None);
            for framebuffer in self.swapchain_framebuffers.iter() {
                self.logical_device.destroy_framebuffer(*framebuffer, None);
            }
            self.logical_device
                .destroy_pipeline(self.graphics_pipeline, None);
            self.logical_device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
        unsafe {
            self.logical_device
                .destroy_render_pass(self.render_pass, None);
        }
        for image_view in self.swapchain_image_views.iter() {
            unsafe {
                self.logical_device.destroy_image_view(*image_view, None);
            }
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }

        unsafe {
            self.logical_device.destroy_device(None);
        }

        println!("Cleaning Up Vulkan Renderer");
        if let Some(debug_utils_messenger) = self.debug_utils_messenger {
            let debug_utils_instance = debug_utils::Instance::new(&self.entry, &self.instance);
            unsafe {
                debug_utils_instance.destroy_debug_utils_messenger(debug_utils_messenger, None)
            };
        }
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
        }
        unsafe { self.instance.destroy_instance(None) };
    }
}
