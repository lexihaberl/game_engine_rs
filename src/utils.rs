use std::ffi::c_char;
use std::ffi::CString;
use std::io::Read;

pub fn vec_cstring_to_vec_pointers(vector: &[CString]) -> Vec<*const c_char> {
    let vec_raw: Vec<*const c_char> = vector
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();
    vec_raw
}

pub fn vec_str_ref_to_vec_cstring(vector: &[&str]) -> Vec<CString> {
    let vec_raw = vector
        .iter()
        .map(|layer| CString::new(*layer).expect("Could not convert str to CString"))
        .collect();
    vec_raw
}

pub fn vec_str_ref_to_vec_pointers(vector: &[&str]) -> (Vec<CString>, Vec<*const c_char>) {
    let vec_cstring = vec_str_ref_to_vec_cstring(vector);
    let vec_pointers = vec_cstring_to_vec_pointers(&vec_cstring);
    // Have to return both so that cstring won't be deallocated which would invalidate pointers
    (vec_cstring, vec_pointers)
}

pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

pub fn read_shader_file(path: &str) -> Vec<u8> {
    std::fs::File::open(path)
        .expect("Failed to open file {path}")
        .bytes()
        .map(|byte| byte.expect("Failed parsing byte code"))
        .collect()
}
