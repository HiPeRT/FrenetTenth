cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(frenet_path_planner LANGUAGES CUDA CXX)

add_subdirectory(include/json)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_LIBRARIES -ldl)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75 -gencode arch=compute_72,code=sm_72")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O2 -mfp16-format")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O2")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS}  -Wall -DDEBUG=1 -O0 -g -mfp16-format")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -DDEBUG=1 -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DDEBUG=1 -O0 -g -G")

include_directories(
  include
  include/json/include
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_executable(measuring
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_link_libraries(measuring PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_double
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_double PRIVATE FRENET_TYPE=CPU prec_type_t=double_prec)
target_link_libraries(measuring_experiments_cpu_double PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_double PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_float
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_float PRIVATE FRENET_TYPE=CPU prec_type_t=float_prec)
target_link_libraries(measuring_experiments_cpu_float PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_float PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_half
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_half PRIVATE FRENET_TYPE=CPU prec_type_t=half_prec)
target_link_libraries(measuring_experiments_cpu_half PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_half PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_double
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_double PRIVATE FRENET_TYPE=GPU prec_type_t=double_prec)
target_link_libraries(measuring_experiments_gpu_double PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_double PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_float
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_float PRIVATE FRENET_TYPE=GPU prec_type_t=float_prec)
target_link_libraries(measuring_experiments_gpu_float PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_float PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_half
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_half PRIVATE FRENET_TYPE=GPU prec_type_t=half_prec)
target_link_libraries(measuring_experiments_gpu_half PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_half PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_double_task
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_double_task PRIVATE FRENET_TYPE=CPU prec_type_t=double_prec MEASURE_TASK_TIME)
target_link_libraries(measuring_experiments_cpu_double_task PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_double_task PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_float_task
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_float_task PRIVATE FRENET_TYPE=CPU prec_type_t=float_prec MEASURE_TASK_TIME)
target_link_libraries(measuring_experiments_cpu_float_task PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_float_task PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_half_task
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_half_task PRIVATE FRENET_TYPE=CPU prec_type_t=half_prec MEASURE_TASK_TIME)
target_link_libraries(measuring_experiments_cpu_half_task PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_half_task PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_double_task
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_double_task PRIVATE FRENET_TYPE=GPU prec_type_t=double_prec MEASURE_TASK_TIME)
target_link_libraries(measuring_experiments_gpu_double_task PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_double_task PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_float_task
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_float_task PRIVATE FRENET_TYPE=GPU prec_type_t=float_prec MEASURE_TASK_TIME)
target_link_libraries(measuring_experiments_gpu_float_task PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_float_task PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_half_task
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_half_task PRIVATE FRENET_TYPE=GPU prec_type_t=half_prec MEASURE_TASK_TIME)
target_link_libraries(measuring_experiments_gpu_half_task PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_half_task PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

#traj

add_executable(measuring_experiments_cpu_double_traj
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments_traj.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_double_traj PRIVATE FRENET_TYPE=CPU prec_type_t=double_prec)
target_link_libraries(measuring_experiments_cpu_double_traj PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_double_traj PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_float_traj
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments_traj.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_float_traj PRIVATE FRENET_TYPE=CPU prec_type_t=float_prec)
target_link_libraries(measuring_experiments_cpu_float_traj PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_float_traj PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_cpu_half_traj
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments_traj.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_cpu_half_traj PRIVATE FRENET_TYPE=CPU prec_type_t=half_prec)
target_link_libraries(measuring_experiments_cpu_half_traj PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_cpu_half_traj PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_double_traj
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments_traj.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_double_traj PRIVATE FRENET_TYPE=GPU prec_type_t=double_prec)
target_link_libraries(measuring_experiments_gpu_double_traj PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_double_traj PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_float_traj
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments_traj.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_float_traj PRIVATE FRENET_TYPE=GPU prec_type_t=float_prec)
target_link_libraries(measuring_experiments_gpu_float_traj PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_float_traj PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(measuring_experiments_gpu_half_traj
        src/frenet_optimal_trajectory_cuda.cu
        src/measuring_experiments_traj.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(measuring_experiments_gpu_half_traj PRIVATE FRENET_TYPE=GPU prec_type_t=half_prec)
target_link_libraries(measuring_experiments_gpu_half_traj PRIVATE ${LIBRARIES} cublas)
target_include_directories(measuring_experiments_gpu_half_traj PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


find_package(Python COMPONENTS Development)
add_executable(test_
        src/frenet_optimal_trajectory.cpp
        src/test.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp)
target_compile_definitions(test_ PRIVATE FRENET_TYPE=CPU)
target_link_libraries(test_ PRIVATE ${LIBRARIES} ${Python_LIBRARIES})
target_include_directories(test_ PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Python_INCLUDE_DIRS})

add_executable(test_gpu_double
        src/frenet_optimal_trajectory_cuda.cu
        src/test.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(test_gpu_double PRIVATE FRENET_TYPE=GPU prec_type_t=double_prec)
target_link_libraries(test_gpu_double PRIVATE ${LIBRARIES} cublas ${Python_LIBRARIES})
target_include_directories(test_gpu_double PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Python_INCLUDE_DIRS})

add_executable(test_gpu_float
        src/frenet_optimal_trajectory_cuda.cu
        src/test.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(test_gpu_float PRIVATE FRENET_TYPE=GPU prec_type_t=float_prec)
target_link_libraries(test_gpu_float PRIVATE ${LIBRARIES} cublas ${Python_LIBRARIES})
target_include_directories(test_gpu_float PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Python_INCLUDE_DIRS})

add_executable(test_gpu_half
        src/frenet_optimal_trajectory_cuda.cu
        src/test.cpp
        src/cubic_spline_planner.cpp
        src/polynomials.cpp
        src/frenet_optimal_trajectory.cpp
        src/frenet_optimal_trajectory_cuda.cpp)
target_compile_definitions(test_gpu_half PRIVATE FRENET_TYPE=GPU prec_type_t=half_prec)
target_link_libraries(test_gpu_half PRIVATE ${LIBRARIES} cublas ${Python_LIBRARIES})
target_include_directories(test_gpu_half PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Python_INCLUDE_DIRS})


