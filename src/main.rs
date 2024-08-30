use std::error::Error;
use std::ffi::CString;
use std::ptr;
use std::thread;
use std::time;

use ash::{vk, Entry};
use winit::application::ApplicationHandler;
use winit::event::ElementState;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};

const WINDOW_WIDTH: f64 = 1280.0;
const WINDOW_HEIGHT: f64 = 720.0;
const WINDOW_NAME: &str = "Vulkan Renderer";
const ENGINE_NAME: &str = "LexEngine";
const VK_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
const APPLICATION_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
const ENGINE_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);

struct VulkanRenderer {
    entry: Entry, //we aren't allowed to call any Vulkan functions after entry is dropped!
    instance: ash::Instance,
}

impl VulkanRenderer {
    fn new(window: &Window) -> Result<VulkanRenderer, Box<dyn Error>> {
        let entry = unsafe { Entry::load()? };

        let instance = VulkanRenderer::create_instance(window, &entry)?;
        Ok(VulkanRenderer { entry, instance })
    }

    fn create_instance(window: &Window, entry: &Entry) -> Result<ash::Instance, Box<dyn Error>> {
        let app_name = CString::new(WINDOW_NAME)
            .expect("Using const rust stringslice. Should never have 0 byte in it");
        let engine_name = CString::new(ENGINE_NAME)
            .expect("Using const rust stringslice. Should never have 0 byte in it");
        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: APPLICATION_VERSION,
            p_engine_name: engine_name.as_ptr(),
            engine_version: ENGINE_VERSION,
            api_version: VK_API_VERSION,
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

    fn draw(&self) {}
    fn swap_buffers(&self) {}
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        println!("Cleaning Up Vulkan Renderer");
        unsafe { self.instance.destroy_instance(None) };
    }
}

struct App {
    window: Option<Window>,
    renderer: Option<VulkanRenderer>,
}

impl App {
    fn new() -> App {
        App {
            window: None,
            renderer: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(WINDOW_NAME)
                    .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT)),
            )
            .expect("Window creation failed");
        self.renderer =
            Some(VulkanRenderer::new(&window).expect(
                "Vulkan initialization failed. Make sure that Vulkan drivers are installed",
            ));
        self.window = Some(window);
        println!("succesfully created window and renderer");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if let (Some(renderer), Some(window)) = (&self.renderer, &self.window) {
            match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    println!("Drawing");
                    renderer.draw();
                    //window.pre_present_notify();
                    thread::sleep(time::Duration::from_millis(500));
                    renderer.swap_buffers();
                    event_loop.exit();
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: key,
                            state: ElementState::Released,
                            ..
                        },
                    ..
                } => match key {
                    PhysicalKey::Code(KeyCode::Escape) => {
                        println!("Escape was pressed; Closing window");
                        event_loop.exit();
                    }
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        println!("Pressing W")
                    }
                    _ => println!("Something else was pressed"),
                },
                _ => (),
            }
        }
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        match cause {
            winit::event::StartCause::Poll => {
                if let Some(window) = &self.window {
                    window.request_redraw();
                    //println!("Poll");
                }
            }
            _ => println!("Ignoring cause: {:?}", cause),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();

    event_loop.run_app(&mut app).expect("Dunno");
    println!("Exiting Program");
}
