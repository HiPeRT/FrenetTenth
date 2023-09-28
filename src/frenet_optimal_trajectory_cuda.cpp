#include <frenet_optimal_trajectory_cuda.hpp>

template class FrenetPlannerGPU<double>;
template class FrenetPlannerGPU<float>;
template class FrenetPlannerGPU<__half>;

template <class T>
FrenetPlannerGPU<T>::~FrenetPlannerGPU(){
    cudaFree(fpp_array_d);
    cudaFree(ob_array_d);

    cudaFreeHost(fpp_array_h);
    cudaFreeHost(ob_array_h);

    cudaFree(spline_x_p);
    cudaFree(spline_y_p);
    cudaFree(spline_ax_p);
    cudaFree(spline_ay_p);
    cudaFree(spline_bx_p);
    cudaFree(spline_by_p);
    cudaFree(spline_cx_p);
    cudaFree(spline_cy_p);
    cudaFree(spline_dx_p);
    cudaFree(spline_dy_p);

    cudaStreamDestroy(stream);
    cudaStreamDestroy(streamObstacles);
    cudaEventDestroy(obstacleCopyComplete);

#ifdef MEASURE_TASK_TIME
    cudaEventDestroy(calc_start);
            cudaEventDestroy(calc_end);
            cudaEventDestroy(check_start);
            cudaEventDestroy(check_end);
#endif

    cublasDestroy(handle);
};

template <class T>
FrenetPlannerGPU<T>::FrenetPlannerGPU(Params<T> params, Spline2D reference_path, Spline2D i_border, Spline2D o_border)
        :params_(params),
         reference_path_(reference_path),
         i_border_(i_border),
         o_border_(o_border){
    minV = params_.target_speed - params_.d_t_s*params_.n_s_sample;
    maxV = params_.target_speed + params_.d_t_s*params_.n_s_sample;

    int blockx = (params_.max_road_width_left - (-params_.max_road_width_right)) / params_.d_road_w;
    int blocky = (params_.maxt - params_.mint + params_.dt)  / params_.dt;
    int blockz = (maxV + params_.d_t_s - minV + params_.d_t_s ) / params_.d_t_s;
    T Ti = params_.maxt;
    int threads = floor(Ti / params_.dt) + 2;

    block = dim3(blockx, blocky, blockz);
    thread = dim3(threads, 1, 1);

    cudaStreamCreate(&stream);
    cudaStreamCreate(&streamObstacles);

    cudaEventCreateWithFlags(&obstacleCopyComplete, cudaEventDisableTiming);

#ifdef MEASURE_TASK_TIME
    cudaEventCreate(&calc_start);
            cudaEventCreate(&calc_end);
            cudaEventCreate(&check_start);
            cudaEventCreate(&check_end);
#endif

    //thread and block fot check_path
    N_PATHS = blockx*blocky*blockz;
    N_POINTS = threads;
    TOTAL_POINTS = N_PATHS * N_POINTS;
    POINTS_TOTAL_SIZE = sizeof(FrenetPathPoint<T>)*TOTAL_POINTS;
    spline_size = reference_path_.sx.x.size();
    last_s_res = reference_path_.get_s_last();

    err = cudaMallocHost((void**)&fpp_array_h, sizeof(FrenetPathPoint<T>)*N_POINTS*3);
    cudaMalloc((void**)&fpp_array_d, POINTS_TOTAL_SIZE);

    OB_MAX_N = params_.next_s_borders > 0 ? params_.next_s_borders*4 : 1024;
    OB_MAX_SIZE = sizeof(obstacle)*OB_MAX_N;
    cudaMalloc((void**)&ob_array_d, OB_MAX_SIZE);
    err = cudaMallocHost((void**)&ob_array_h, OB_MAX_SIZE);

    THREADSX = 16;
    THREADSY = 16;
    THREADSZ = 4;
    float POINTS_PER_BLOCK = THREADSX*THREADSY*THREADSZ;
    int BLOCKS = ceil(TOTAL_POINTS/POINTS_PER_BLOCK);
    BLOCKSX = ceil(sqrt(BLOCKS));
    BLOCKSY = BLOCKSX;

    cudaMalloc((void**)&spline_x_p,  sizeof(double)* reference_path_.sx.x.size());
    cudaMalloc((void**)&spline_y_p,  sizeof(double)* reference_path_.sy.x.size());
    cudaMalloc((void**)&spline_ax_p, sizeof(double)*reference_path_.sx.a.size());
    cudaMalloc((void**)&spline_ay_p, sizeof(double)*reference_path_.sy.a.size());
    cudaMalloc((void**)&spline_bx_p, sizeof(double)*reference_path_.sx.b.size());
    cudaMalloc((void**)&spline_by_p, sizeof(double)*reference_path_.sy.b.size());
    cudaMalloc((void**)&spline_cx_p, sizeof(double)*reference_path_.sx.c.size());
    cudaMalloc((void**)&spline_cy_p, sizeof(double)*reference_path_.sy.c.size());
    cudaMalloc((void**)&spline_dx_p, sizeof(double)*reference_path_.sx.d.size());
    cudaMalloc((void**)&spline_dy_p, sizeof(double)*reference_path_.sy.d.size());

    cudaMemcpyAsync(spline_x_p,  reference_path_.sx.x.data(), sizeof(double)*reference_path_.sx.x.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_y_p,  reference_path_.sy.x.data(), sizeof(double)*reference_path_.sy.x.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_ax_p, reference_path_.sx.a.data(), sizeof(double)*reference_path_.sx.a.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_ay_p, reference_path_.sy.a.data(), sizeof(double)*reference_path_.sy.a.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_bx_p, reference_path_.sx.b.data(), sizeof(double)*reference_path_.sx.b.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_by_p, reference_path_.sy.b.data(), sizeof(double)*reference_path_.sy.b.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_cx_p, reference_path_.sx.c.data(), sizeof(double)*reference_path_.sx.c.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_cy_p, reference_path_.sy.c.data(), sizeof(double)*reference_path_.sy.c.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_dx_p, reference_path_.sx.d.data(), sizeof(double)*reference_path_.sx.d.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(spline_dy_p, reference_path_.sy.d.data(), sizeof(double)*reference_path_.sy.d.size(), cudaMemcpyHostToDevice, stream);

    init();

    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    //err = cudaPeekAtLastError();

};
