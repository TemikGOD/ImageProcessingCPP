# CMakeList.txt : CMake project for Lab1, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (Lab1 "Lab1.cpp" "Lab1.h" "adImgPro.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Lab1 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

# set_target_properties(Lab1 PROPERTIES
# 	CUDA_SEPARABLE_COMPILATION ON
# )

# find_package(CUDAToolkit REQUIRED)
# include_directories(${CUDAToolkit_INCLUDE_DIRS})
# target_link_libraries(Lab1 CUDA::cudart)
# find_package(wxWidgets REQUIRED COMPONENTS net core base)
# include(${wxWidgets_USE_FILE})
# target_link_libraries(Lab1 ${wxWidgets_LIBRARIES})
find_package(OpenCV REQUIRED)
target_link_libraries(Lab1 ${OpenCV_LIBS})