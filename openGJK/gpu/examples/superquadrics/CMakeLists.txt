project(example_lib_openGJK_GPU_superquadrics
    LANGUAGES C CXX CUDA
    VERSION 1.0.0
)

add_executable(example_lib_openGJK_GPU_superquadrics
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
)

target_link_libraries(example_lib_openGJK_GPU_superquadrics openGJK_GPU)

# CUDA architecture handling (same as parent)
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(example_lib_openGJK_GPU_superquadrics PROPERTIES
        CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(example_lib_openGJK_GPU_superquadrics PROPERTIES
        CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(example_lib_openGJK_GPU_superquadrics PROPERTIES
        CUDA_ARCHITECTURES native)
endif()
