use ash::vk;

pub const WINDOW_WIDTH: f64 = 1280.0;
pub const WINDOW_HEIGHT: f64 = 720.0;
pub const WINDOW_NAME: &str = "Vulkan Renderer";
pub const ENGINE_NAME: &str = "LexEngine";
pub const VK_API_VERSION: u32 = vk::make_api_version(0, 1, 3, 0);
pub const APPLICATION_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
pub const ENGINE_VERSION: u32 = vk::make_api_version(0, 0, 1, 0);
pub const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
pub const REQUIRED_DEVICE_EXTENSIONS: [&str; 1] = ["VK_KHR_swapchain"];
pub const DYNAMIC_STATE: [vk::DynamicState; 2] =
    [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
