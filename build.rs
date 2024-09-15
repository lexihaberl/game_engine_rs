use std::process::Command;

fn main() {
    let script = "./compile_shaders.sh";

    let status = Command::new("sh")
        .arg(script)
        .status()
        .expect("Failed to execute shader compilation script");

    if !status.success() {
        panic!("Shader compilation failed!");
    }
}
