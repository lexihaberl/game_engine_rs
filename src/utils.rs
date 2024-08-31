use std::ffi::c_char;
use std::ffi::CString;

pub fn vec_cstring_to_vec_pointers(vector: &Vec<CString>) -> Vec<*const c_char> {
    let vec_raw: Vec<*const c_char> = vector
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();
    vec_raw
}
