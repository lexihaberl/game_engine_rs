use crate::config;
use ash::{vk, Entry};
use std::error::Error;
use std::ffi::CString;
use std::ptr;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::Window;

pub struct VulkanRenderer {
    entry: Entry, //we aren't allowed to call any Vulkan functions after entry is dropped!
    instance: ash::Instance,
}

impl VulkanRenderer {
    pub fn new(window: &Window) -> Result<VulkanRenderer, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        let instance = VulkanRenderer::create_instance(window, &entry)?;
        Ok(VulkanRenderer { entry, instance })
    }

    fn create_instance(window: &Window, entry: &Entry) -> Result<ash::Instance, Box<dyn Error>> {
        if cfg!(debug_assertions) {
            println!("Vulkan validation layers enabled!");
        } else {
            println!("Vulkan validation layers disabled!");
        }
        let app_name = CString::new(config::WINDOW_NAME)
            .expect("Using const rust stringslice. Should never have 0 byte in it");
        let engine_name = CString::new(config::ENGINE_NAME)
            .expect("Using const rust stringslice. Should never have 0 byte in it");
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
        let extension_names =
            ash_window::enumerate_required_extensions(window.display_handle()?.as_raw())?;
        let create_info = vk::InstanceCreateInfo {
            s_type: vk::StructureType::INSTANCE_CREATE_INFO,
            p_application_info: &app_info,
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            p_next: ptr::null(),
            flags: vk::InstanceCreateFlags::empty(),
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            ..Default::default()
        };

        // safety: needs entry to be valid, which is ensured by reference
        let instance = unsafe { entry.create_instance(&create_info, Option::None)? };
        Ok(instance)
    }

    pub fn draw(&self) {}
    pub fn swap_buffers(&self) {}
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        println!("Cleaning Up Vulkan Renderer");
        unsafe { self.instance.destroy_instance(None) };
    }
}
