use crate::config;
use crate::utils;
use ash::ext::debug_utils;
use ash::{vk, Entry};
use core::panic;
use std::cmp::Reverse;
use std::error::Error;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use winit::raw_window_handle::HasDisplayHandle;
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
    physical_device: vk::PhysicalDevice,
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
        let physical_device = Self::get_physical_device(&instance);
        Ok(VulkanRenderer {
            entry,
            instance,
            debug_utils_messenger,
            physical_device,
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

    fn get_physical_device(instance: &ash::Instance) -> vk::PhysicalDevice {
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
            .filter(|device| Self::is_device_suitable(instance, *device))
            .collect();
        println!("Found {} suitable devices", suitable_physical_devices.len());

        suitable_physical_devices
            .sort_by_key(|device| Reverse(Self::get_device_suitability_score(instance, *device)));

        if suitable_physical_devices.is_empty() {
            panic!("No suitable devices found!")
        }

        let device_properties =
            unsafe { instance.get_physical_device_properties(suitable_physical_devices[0]) };
        let device_name = device_properties
            .device_name_as_c_str()
            .expect("Should be able to convert dev name to c_str");
        println!("Choosing device {:?}", device_name);
        suitable_physical_devices[0]
    }

    fn is_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        for queue_family_property in queue_family_properties.iter() {
            if queue_family_property
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
            {
                return true;
            }
        }
        false
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

    pub fn draw(&self) {}
    pub fn swap_buffers(&self) {}
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        println!("Cleaning Up Vulkan Renderer");
        if let Some(debug_utils_messenger) = self.debug_utils_messenger {
            let debug_utils_instance = debug_utils::Instance::new(&self.entry, &self.instance);
            unsafe {
                debug_utils_instance.destroy_debug_utils_messenger(debug_utils_messenger, None)
            };
        }
        unsafe { self.instance.destroy_instance(None) };
    }
}
