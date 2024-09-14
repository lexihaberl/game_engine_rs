use game_engine::config;
use game_engine::VulkanRenderer;
use winit::application::ApplicationHandler;
use winit::event::ElementState;
use winit::event::{KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

struct App {
    renderer: Option<VulkanRenderer>,
    window: Option<Window>,
    last_frame: std::time::Instant,
}

impl App {
    fn new() -> App {
        App {
            // Rust fields are dropped in order => renderer has to be dropped before window!
            // Therefore, renderer is placed before window in the struct
            renderer: None,
            window: None,
            last_frame: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(config::WINDOW_NAME)
                    .with_inner_size(winit::dpi::LogicalSize::new(
                        config::WINDOW_WIDTH,
                        config::WINDOW_HEIGHT,
                    )),
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
        if let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) {
            let mut exit = false;
            match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    exit = true;
                }
                WindowEvent::RedrawRequested => {
                    self.last_frame = std::time::Instant::now();
                    window.pre_present_notify();
                    renderer.draw();
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
                        exit = true;
                    }
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        println!("Pressing W")
                    }
                    _ => println!("Something else was pressed"),
                },
                _ => (),
            }
            if exit {
                event_loop.exit();
                renderer.wait_idle();
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

    event_loop
        .run_app(&mut app)
        .expect("Runtime Error in the eventloop");
    println!("Exiting Program");
}
