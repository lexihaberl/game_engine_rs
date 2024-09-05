mkdir -p shaders_compiled
glslc shaders/triangle.frag -o shaders_compiled/triangle_frag.spv
glslc shaders/triangle.vert -o shaders_compiled/triangle_vert.spv
