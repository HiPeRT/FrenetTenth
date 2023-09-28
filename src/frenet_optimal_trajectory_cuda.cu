#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cubic_spline_planner.hpp>
#include <cublas_v2.h>
#include <float.h>
#include <cuda_fp16.hpp>
#include "../include/frenet_optimal_trajectory_cuda.hpp"

#if prec_type_t == half_prec
	#define PREC_MAX FLT_MAX
#elif prec_type_t == float_prec
	#define PREC_MAX FLT_MAX
#elif prec_type_t == double_prec
	#define PREC_MAX DBL_MAX
#endif


template class FrenetPlannerGPU<double>;
template class FrenetPlannerGPU<float>;
template class FrenetPlannerGPU<__half>;


inline cublasStatus_t cublasAmin(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
    return cublasIdamin(handle, n, x, incx, result);
}

inline cublasStatus_t cublasAmin(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
    return cublasIsamin(handle, n, x, incx, result);
}

inline cublasStatus_t cublasAmin(cublasHandle_t handle, int n, const __half *x, int incx, int *result) {
    return cublasIsamin(handle, n, (float *)x, incx, result);
}

template <class T>
__device__ __constant__ ParamsGPU<T> params_gpu;

__device__ __constant__ double *spline_x;
__device__ __constant__ double *spline_y;
__device__ __constant__ double *spline_ax;
__device__ __constant__ double *spline_ay;
__device__ __constant__ double *spline_bx;
__device__ __constant__ double *spline_by;
__device__ __constant__ double *spline_cx;
__device__ __constant__ double *spline_cy;
__device__ __constant__ double *spline_dx;
__device__ __constant__ double *spline_dy;

__device__ __constant__ int num_points;
__device__ __constant__ int total_points_gpu;

template <class T>
__device__ __constant__ FrenetPathPoint<T> *fpp_array;
__device__ __constant__ obstacle *ob_array;

__device__ __constant__ int spline_size_gpu;

template <class T>
__device__ __constant__ T last_s;

template <class T>
__device__ __constant__ T minV_gpu;
template <class T>
__device__ __constant__ T maxV_gpu;


template <class T>
struct quintic{
	T xs, vxs, axs, xe, vxe, axe, a0, a1, a2, a3, a4, a5;
};
template <class T>
__device__ void quintic_init(quintic<T> &q, T, T, T, T, T, T, T);
template <class T>
__device__ T calc_point(quintic<T> &q, T);
template <class T>
__device__ T calc_first_derivative(quintic<T> &q, T);
template <class T>
__device__ T calc_second_derivative(quintic<T> &q, T);
template <class T>
__device__ T calc_third_derivative(quintic<T> &q, T);

template <class T>
struct quartic{
	T xs, vxs, axs, vxe, axe, a0, a1, a2, a3, a4;
};
template <class T>
__device__ void quartic_init(quartic<T> &q, T, T, T, T, T, T);
template <class T>
__device__ T calc_point(quartic<T> &q, T);
template <class T>
__device__ T calc_first_derivative(quartic<T> &q, T);
template <class T>
__device__ T calc_second_derivative(quartic<T> &q, T);
template <class T>
__device__ T calc_third_derivative(quartic<T> &q, T);

template <class T>
__device__ T* upper_bound(T *start, T *end, T value);

template <class T>
__inline__ __device__ T dist(T, T, T, T);

__device__ inline double _sin(double x) {
    return sin(x);
}
__device__ inline float _sin(float x) {
    return sin(x);
}
__device__ inline __half _sin(__half x) {
    return hsin(x);
}

__device__ inline double _cos(double x) {
    return cos(x);
}
__device__ inline float _cos(float x) {
    return cos(x);
}
__device__ inline __half _cos(__half x) {
    return hcos(x);
}

__device__ inline double _sqrt(double x) {
    return sqrt(x);
}
__device__ inline float _sqrt(float x) {
    return sqrt(x);
}
__device__ inline __half _sqrt(__half x) {
    return hsqrt(x);
}

__device__ inline double _atan2(double x, double y) {
    return atan2(x, y);
}
__device__ inline float _atan2(float x, float y) {
    return atan2f(x, y);
}
__device__ inline __half _atan2(__half x, __half y) {
    return atan2f(x, y);
}

__device__ inline int _round(double x) {
    return round(x);
}
__device__ inline int _round(float x) {
    return round(x);
}
__device__ inline int _round(__half x) {
    return __half2int_rn(x);
}

/**
Kernel that check the path considering the obstacles. On block.z axis there is the obstacles
*/
template <class T>
__global__ void check_paths_kernel(){
	int threads_per_block = (blockDim.x*blockDim.y*blockDim.z);
	int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
	int fpp_index = blockId*threads_per_block + ((threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
	
	//if index is higher that the total of points it can die
	if (fpp_index >= total_points_gpu){
		return;
	}

	int path_index = fpp_index / num_points;
	int point_index = fpp_index - path_index * num_points;
	int ob_index = blockIdx.z;

	int first_point_path_index = path_index * num_points;

	//if the piont is away from path range chech thread can die
	if ((T)point_index > (T)(fpp_array<T>[first_point_path_index].length)*(T)(params_gpu<T>.range_path_check)){
		return;
	}

	//if this path is already invalid thread can die
	if (fpp_array<T>[first_point_path_index].obstacle == 1){
		return;
	}	

	T px = (T)fpp_array<T>[fpp_index].x;
	T py = (T)fpp_array<T>[fpp_index].y;
	T ox = (T)ob_array[ob_index].x;
	T oy = (T)ob_array[ob_index].y;
	T orad = (T)ob_array[ob_index].radius;
	T di = dist<T>(px, py, ox, oy) - orad;

	bool collision = di < (T)params_gpu<T>.safe_distance;
	if (collision) {
		fpp_array<T>[first_point_path_index].obstacle = 1;
		fpp_array<T>[first_point_path_index].cost_ = PREC_MAX;
	}

}

/**
Kernel that compute the frenet paths each dimension have a semantic:
block.x -> di
block.y -> Ti
block.z -> tv
thread.x -> t
*/
template <class T>
__global__ void
__launch_bounds__(1024, 1)
calc_frenet_paths_kernel(T s_d, T c_d, T c_d_d, T c_d_dd, T s0){
	const T di = -(T)params_gpu<T>.max_road_width_right + ((T)params_gpu<T>.d_road_w * (T)blockIdx.x);
	const T Ti = (T)params_gpu<T>.mint + ((T)params_gpu<T>.dt * (T)blockIdx.y);
	const T tv = minV_gpu<T> + ((T)params_gpu<T>.d_t_s * (T)blockIdx.z);
	const T t = (T)params_gpu<T>.dt * (T)threadIdx.x;

	const int length = _round((Ti / params_gpu<T>.dt) + (T)2);
	if (threadIdx.x >= length){
		return;
	}

	//quintic and quartic init
	struct quintic<T> lat_qp;
	quintic_init<T>(lat_qp, c_d, c_d_d, c_d_dd, di, (T)0.0, (T)0.0, Ti);
	struct quartic<T> lon_qp;
	quartic_init<T>(lon_qp, s0, s_d, (T)0.0, tv, (T)0.0, Ti);

	const int fp_index = (blockIdx.z *gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x) + blockIdx.x;
	const int point_index = threadIdx.x;

	const int fpp_index = (fp_index * num_points) + point_index;

	// calc the value
	const T s = calc_point<T>(lon_qp, t);
	const T d = calc_point<T>(lat_qp, t);
	const T d_ddd = calc_third_derivative<T>(lat_qp, t);
	const T s_ddd = calc_third_derivative<T>(lon_qp, t);
	fpp_array<T>[fpp_index].t = t;
	fpp_array<T>[fpp_index].Ti = Ti;
	fpp_array<T>[fpp_index].length = length;
	fpp_array<T>[fpp_index].d = d;
	fpp_array<T>[fpp_index].d_d = calc_first_derivative<T>(lat_qp, t);
	fpp_array<T>[fpp_index].d_dd = calc_second_derivative<T>(lat_qp, t);
	fpp_array<T>[fpp_index].d_ddd = d_ddd;
	fpp_array<T>[fpp_index].s = s;
	fpp_array<T>[fpp_index].s_d = calc_first_derivative<T>(lon_qp, t);
	fpp_array<T>[fpp_index].s_dd = calc_second_derivative<T>(lon_qp, t);
	fpp_array<T>[fpp_index].s_ddd = s_ddd;

	//partial points cost;
	extern __shared__ double JpJs_partials[];
	T *Jp_partials = (T*)(&JpJs_partials);
    T *Js_partials = (T*)Jp_partials + (blockDim.x);
	Jp_partials[threadIdx.x] = d_ddd * d_ddd;
	Js_partials[threadIdx.x] = s_ddd * s_ddd;

	//calc of global yaw x and y
	T new_s = s / (T)last_s<T>;
	const int laps = (int)new_s;
	new_s = s - (T)last_s<T>*(T)laps;
	const int ix = upper_bound<double>(&(spline_x[0]), &(spline_x[spline_size_gpu-1]), (double)new_s) - &(spline_x[0]) - 1;
	const int iy = upper_bound<double>(&(spline_y[0]), &(spline_y[spline_size_gpu-1]), (double)new_s) - &(spline_y[0]) - 1;
	const T dx = new_s - (T)spline_x[ix];
	const T dy = new_s - (T)spline_y[iy];
	const T ixv = (T)spline_ax[ix] + (T)spline_bx[ix]*dx + (T)spline_cx[ix]*dx*dx + (T)spline_dx[ix]*dx*dx*dx;
	const T iyv = (T)spline_ay[iy] + (T)spline_by[iy]*dy + (T)spline_cy[iy]*dy*dy + (T)spline_dy[iy]*dy*dy*dy;
	const T yaw_dx = (T)spline_bx[ix] + (T)2*(T)spline_cx[ix]*dx + (T)3*(T)spline_dx[ix]*dx*dx;
	const T yaw_dy = (T)spline_by[iy] + (T)2*(T)spline_cy[iy]*dy + (T)3*(T)spline_dy[iy]*dy*dy;
	const T yawxy = _atan2(yaw_dy, yaw_dx);

	const T x = ixv - d*_sin(yawxy);
	const T y = iyv + d*_cos(yawxy);
	fpp_array<T>[fpp_index].x = x;
	fpp_array<T>[fpp_index].y = y;
	__syncthreads();

	const int next_index = fpp_index + 1 - ((point_index + 1)==length);
	const T dxx = (T)fpp_array<T>[next_index].x - x;
	const T dyy = (T)fpp_array<T>[next_index].y - y;
	const T yaw = _atan2(dyy, dxx);
	const T ds = _sqrt(dxx*dxx + dyy*dyy);
	fpp_array<T>[fpp_index].yaw = yaw;
	fpp_array<T>[fpp_index].ds = ds;
	__syncthreads();

	const T c = (T)fpp_array<T>[next_index].yaw - yaw / ds;
	fpp_array<T>[fpp_index].c = c;

	// cumulative costs
	if (fpp_index % num_points != 0){
		return;
	}
	int length_int = length;
	T Jp = 0;
	T Js = 0;
	for (int i=0; i<length_int; i++){
		Jp += Jp_partials[i];
		Js += Js_partials[i];
	}
	T ds_cost = params_gpu<T>.target_speed - fpp_array<T>[fpp_index+length_int-1].s_d;
	ds_cost = ds_cost * ds_cost;
	const T cd = (T)params_gpu<T>.kj*Jp + (T)params_gpu<T>.kt*Ti + (T)params_gpu<T>.kd*(T)fpp_array<T>[fpp_index+length_int-1].d*(T)fpp_array<T>[fpp_index+length_int-1].d;
	const T cv = (T)params_gpu<T>.kj*Js + (T)params_gpu<T>.kt*Ti + (T)params_gpu<T>.kd*ds_cost;
	fpp_array<T>[fpp_index].cd = cd;
	fpp_array<T>[fpp_index].cv = cv;
	const T cf = (T)params_gpu<T>.klat*cd + (T)params_gpu<T>.klon*cv;
	fpp_array<T>[fpp_index].cf = cf;
	fpp_array<T>[fpp_index].cost_ = cf;

	//init obstacle check for next kernel
	fpp_array<T>[fpp_index].obstacle = 0;

}

/**
Convert the array of FrenetPathPoint into a FrenetPath vector
*/
template <class T>
std::vector<FrenetPath<double>> get_frenet_paths_vector(FrenetPathPoint<T> *array, int num_points, int num_paths){
	std::vector<FrenetPath<double>> fp_vector;
	fp_vector.reserve(num_paths);
	for (int i=0; i<num_paths; i++){
		FrenetPathPoint<T> *path = &(array[i*num_points]);
		FrenetPath<double> fp;
		for (int j=0; j<(int)path->length; j++){
			FrenetPathPoint<T> p = path[j];
			fp.t.push_back(p.t);
			fp.d.push_back(p.d);
			fp.d_d.push_back(p.d_d);
			fp.d_dd.push_back(p.d_dd);
			fp.d_ddd.push_back(p.d_ddd);
			fp.s.push_back(p.s);
			fp.s_d.push_back(p.s_d);
			fp.s_dd.push_back(p.s_dd);
			fp.s_ddd.push_back(p.s_ddd);
			fp.x.push_back(p.x);
			fp.y.push_back(p.y);
			fp.yaw.push_back(p.yaw);
			fp.ds.push_back(p.ds);
			fp.c.push_back(p.c);
		}

		fp.cd = path[0].cd;
		fp.cv = path[0].cv;
		fp.cf = path[0].cf;

		fp_vector.push_back(fp);
	}

	return fp_vector;
}

/**
Inizialization of constants on gpu
*/
template <class T>
void FrenetPlannerGPU<T>::init(){
	err = cudaMemcpyToSymbolAsync(params_gpu<T>, &(params_), sizeof(ParamsGPU<T>), 0, cudaMemcpyHostToDevice, stream);

	err = cudaMemcpyToSymbolAsync(spline_x , &(this->spline_x_p ), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_y , &(this->spline_y_p ), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_ax, &(this->spline_ax_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_ay, &(this->spline_ay_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_bx, &(this->spline_bx_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_by, &(this->spline_by_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_cx, &(this->spline_cx_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_cy, &(this->spline_cy_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_dx, &(this->spline_dx_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(spline_dy, &(this->spline_dy_p), sizeof(double*), 0, cudaMemcpyHostToDevice, stream);

	err = cudaMemcpyToSymbolAsync(num_points, &(this->N_POINTS), sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(total_points_gpu, &(this->TOTAL_POINTS), sizeof(int), 0, cudaMemcpyHostToDevice, stream);

	err = cudaMemcpyToSymbolAsync(fpp_array<T>, &(this->fpp_array_d), sizeof(FrenetPathPoint<T>*), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(ob_array, &(this->ob_array_d), sizeof(obstacle*), 0, cudaMemcpyHostToDevice, stream);

	err = cudaMemcpyToSymbolAsync(spline_size_gpu, &(this->spline_size), sizeof(int), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(last_s<T>, &(this->last_s_res), sizeof(T), 0, cudaMemcpyHostToDevice, stream);

	err = cudaMemcpyToSymbolAsync(minV_gpu<T>, &(this->minV), sizeof(T), 0, cudaMemcpyHostToDevice, stream);
	err = cudaMemcpyToSymbolAsync(maxV_gpu<T>, &(this->maxV), sizeof(T), 0, cudaMemcpyHostToDevice, stream);
}

/**
This function call the kernels for calc of paths and the check
*/
template <class T>
void FrenetPlannerGPU<T>::calc_frenet_paths(double s_d, double c_d, double c_d_d, double c_d_dd, double s0, FrenetPathPoint<T> **ret_array, int *n_paths, int *n_points, int N_OB){
#ifdef MEASURE_TASK_TIME
    cudaEventRecord(calc_start, stream);
#endif
	calc_frenet_paths_kernel<T><<<block, thread, thread.x*sizeof(T)*2, stream>>>(s_d, c_d, c_d_d, c_d_dd, s0);
#ifdef MEASURE_TASK_TIME
    cudaEventRecord(calc_end, stream);
#endif

#ifdef DEBUG
    cudaStreamSynchronize(stream);
	err = cudaPeekAtLastError();
    std::cout << cudaGetErrorString(err) << std::endl;
#endif
	if (N_OB > 0) {
        cudaStreamWaitEvent(stream, obstacleCopyComplete);

        dim3 block_ob(BLOCKSX, BLOCKSY, N_OB);
        dim3 thread_ob(THREADSX, THREADSY, THREADSZ);

        int OB_TOTAL_SIZE = sizeof(obstacle) * N_OB;
        cudaMemcpyAsync(ob_array_d, ob_array_h, OB_TOTAL_SIZE, cudaMemcpyHostToDevice, streamObstacles);
        cudaEventRecord(obstacleCopyComplete, streamObstacles);
#ifdef DEBUG
        cudaStreamSynchronize(stream);
        err = cudaPeekAtLastError();
#endif

#ifdef MEASURE_TASK_TIME
        cudaEventRecord(check_start, stream);
#endif
        check_paths_kernel<T><<<block_ob, thread_ob, 0, stream>>>();
#ifdef MEASURE_TASK_TIME
        cudaEventRecord(check_end, stream);
#endif
#ifdef DEBUG
        cudaStreamSynchronize(stream);
        err = cudaPeekAtLastError();
#endif
    }
    stat = cublasAmin(handle, N_PATHS, (T*)(&(fpp_array_d[0].cost_)), N_POINTS*sizeof(FrenetPathPoint<T>)/sizeof(T), &best_path_index);

#ifdef DEBUG
    cudaStreamSynchronize(stream);
	err = cudaPeekAtLastError();
#endif

    //cudaMemcpyAsync(fpp_array_h, fpp_array_d, sizeof(FrenetPathPoint<T>)*N_POINTS, cudaMemcpyDeviceToHost, stream);
	//cudaMemcpyAsync(&(fpp_array_h[N_POINTS]), &(fpp_array_d[N_POINTS*(N_PATHS-1)]), sizeof(FrenetPathPoint<T>)*N_POINTS, cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(&(fpp_array_h[N_POINTS*2]), &(fpp_array_d[N_POINTS*(best_path_index-1)]), sizeof(FrenetPathPoint<T>)*N_POINTS, cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);
#ifdef DEBUG
	err = cudaPeekAtLastError();
#endif

	*ret_array = fpp_array_h;
	*n_paths = N_PATHS;
	*n_points = N_POINTS;
}

template <class T>
__inline__ __device__ T dist(T x1, T y1, T x2, T y2){
	T dx = x1 - x2;
	T dy = y1 - y2;
	return _sqrt(dx*dx + dy*dy);
}

/**
Function for computing the frenet optimal path
*/
template <class T>
FrenetPath<double> FrenetPlannerGPU<T>::frenet_optimal_planning(double s0, double s_d, double c_d, double c_d_d, double c_d_dd, std::vector<obstacle> obstacles, FrenetPath<double> &first, FrenetPath<double> &last, int overtake_strategy)
{

	switch(overtake_strategy)
	{
		case 1:
			params_.max_road_width_left = params_.max_road_width;
			params_.max_road_width_right = 0;
			break;
		case 2:
			params_.max_road_width_left = 0;
			params_.max_road_width_right = params_.max_road_width;
			break;
		default:
			params_.max_road_width_left = params_.max_road_width;
			params_.max_road_width_right = params_.max_road_width;
			break;
	}

	int OB_SIZE = obstacles.size();
	if (OB_SIZE > OB_MAX_N - 2*params_.next_s_borders){
		printf("Too much obstacles, some may be dropped!");
	}

	memcpy(ob_array_h, obstacles.data(), OB_SIZE*sizeof(obstacle));

	obstacle ob_temp;

    //Add the track borders as obstacles
    double x_i, y_i, x_o, y_o;
    double s_i = s0;
    double s_o = s0;
    if (!i_border_.sx.x.empty() && !o_border_.sx.x.empty()){
        for (int i = 0; i < params_.next_s_borders && OB_SIZE < OB_MAX_N - 2; ++i) {
            s_i += i;
            s_o += i;
            if (s_i > i_border_.get_s_last())
                s_i = 0;

            if (s_o > o_border_.get_s_last())
                s_o = 0;

            i_border_.calc_position(&x_i, &y_i, s_i);
            o_border_.calc_position(&x_o, &y_o, s_o);

            ob_temp = {
                    .x = x_i,
                    .y = y_i,
                    .radius = 1.0,
                    .s = s_i,
                    .d = 0};

            ob_array_h[OB_SIZE++] = ob_temp;

            ob_temp = {
                    .x = x_o,
                    .y = y_o,
                    .radius = 1.0,
                    .s = s_o,
                    .d = 0};

            ob_array_h[OB_SIZE++] = ob_temp;
        }
    }

	FrenetPathPoint<T> *fplist;
	int n_paths, n_points;
    calc_frenet_paths(s_d, c_d, c_d_d, c_d_dd, s0, &fplist, &n_paths, &n_points, OB_SIZE);

	if (fplist[2*n_points].obstacle != 0) {
		FrenetPath<double> res;
		res.empty = true;
		return res;
	}
	auto res = get_frenet_paths_vector<T>(&(fplist[2*n_points]), n_points, 1)[0];

#ifdef MEASURE_TASK_TIME
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, calc_start, calc_end);
    time_calc = milliseconds;
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, check_start, check_end);
    time_check = milliseconds;
#endif

	return res;
}


template <class T>
__inline__ __device__ T* upper_bound(T *start, T* end, T value){
	T *p = start;
	while (value > *p && p < end){
		p++;
	}
	return p;
}

template <class T, int N>
__device__ bool inverse(T A[N][N], T inverse[N][N]);
template <class T>
__device__ bool inverse3(T A[3][3], T inverse[3][3]);
template <class T>
__device__ bool inverse2(T A[2][2], T inverse[2][2]);

template <class T, int N>
__device__ void matrix_x_vector(int n, T y[N], T x[N][N], T A[N]);
template <class T>
__device__ void matrix_x_vector2(T y[2], T x[2][2], T A[2]);
template <class T>
__device__ void matrix_x_vector3(T y[3], T x[3][3], T A[3]);


/**
Quintic implementation on device
*/
template <class C>
__device__ void quintic_init(quintic<C> &q ,C xs_t, C vxs_t, C axs_t, C xe_t, C vxe_t, C axe_t, C T)
{
	q.xs = xs_t;
	q.vxs = vxs_t;
	q.axs = axs_t;
	q.xe = xe_t;
	q.vxe = vxe_t;
	q.axe = axe_t;

	q.a0 = q.xs;
	q.a1 = q.vxs;
	q.a2 = q.axs / (C)2.0;

	double X[3] = {0, 0, 0};
	double A_inv[3][3];

	double A[3][3] = {T*T*T, T*T*T*T, T*T*T*T*T,
	     (C)3*T*T, (C)4*T*T*T, (C)5*T*T*T*T,
	     (C)6*T, (C)12*T*T, (C)20*T*T*T};

	double B[3] = {q.xe - q.a0 - q.a1*T - q.a2*T*T, 
	     q.vxe - q.a1 - (C)2*q.a2*T,
	     q.axe - (C)2*q.a2};

	inverse3<double>(A, A_inv);
	matrix_x_vector<double,3>(3, B, A_inv, X);

	q.a3 = (C)X[0];
	q.a4 = (C)X[1];
	q.a5 = (C)X[2];
}

template <class C>
__inline__ __device__ C calc_point(quintic<C> &q, C t)
{
	C xt = q.a0 + q.a1*t + q.a2*t*t + q.a3*t*t*t + q.a4*t*t*t*t + q.a5*t*t*t*t*t;
	return xt;
}

template <class C>
__inline__ __device__ C calc_first_derivative(quintic<C> &q, C t)
{
	C xt = q.a1 + (C)2*q.a2*t + (C)3*q.a3*t*t + (C)4*q.a4*t*t*t + (C)5*q.a5*t*t*t*t;
	return xt;
}

template <class C>
__inline__ __device__ C calc_second_derivative(quintic<C> &q, C t)
{
	C xt = (C)2*q.a2 + (C)6*q.a3*t + (C)12*q.a4*t*t + (C)20*q.a5*t*t*t;
	return xt;
}

template <class C>
__inline__ __device__ C calc_third_derivative(quintic<C> &q, C t)
{
	C xt = (C)6*q.a3 + (C)24*q.a4*t + (C)60*q.a5*t*t;
	return xt;
}


/**
Quartic implementation on device
*/
template <class C>
__device__ void quartic_init(quartic<C> &q, C xs_t, C vxs_t, C axs_t, C vxe_t, C axe_t, C T)
{
	q.xs = xs_t;
	q.vxs = vxs_t;
	q.axs = axs_t;
	q.vxe = vxe_t;
	q.axe = axe_t;

	q.a0 = q.xs;
	q.a1 = q.vxs;
	q.a2 = q.axs / (C)2.0;

	double X[2] = {0, 0};
	double A_inv[2][2];

	double A[2][2] = {(C)3*T*T, (C)4*T*T*T,
	     (C)6*T, (C)12*T*T};

	double B[2] = {q.vxe - q.a1 - (C)2*q.a2*T,
	     q.axe - (C)2*q.a2};

	inverse<double,2>(A,A_inv);
	matrix_x_vector<double,2>(2, B, A_inv, X);

	q.a3 = (C)X[0];
	q.a4 = (C)X[1];
}

template <class C>
__inline__ __device__ C calc_point(quartic<C> &q, C t)
{
	C xt = q.a0 + q.a1*t + q.a2*t*t + q.a3*t*t*t + q.a4*t*t*t*t;
	return xt;
}

template <class C>
__inline__ __device__ C calc_first_derivative(quartic<C> &q, C t)
{
	C xt = q.a1 + (C)2*q.a2*t + (C)3*q.a3*t*t + (C)4*q.a4*t*t*t;
	return xt;
}

template <class C>
__inline__ __device__ C calc_second_derivative(quartic<C> &q, C t)
{
	C xt = (C)2*q.a2 + (C)6*q.a3*t + (C)12*q.a4*t*t;
	return xt;
}

template <class C>
__inline__ __device__ C calc_third_derivative(quartic<C> &q, C t)
{
	C xt = (C)6*q.a3 + (C)24*q.a4*t;
	return xt;
}


template <class T, int N> 
__device__ void getCofactor(T A[N][N], T temp[N][N], int p, int q, int n) 
{ 
    int i = 0, j = 0; 
  
    // Looping for each element of the matrix 
    for (int row = 0; row < n; row++) 
    { 
        for (int col = 0; col < n; col++) 
        { 
            //  Copying into temporary matrix only those element 
            //  which are not in given row and column 
            if (row != p && col != q) 
            { 
                temp[i][j++] = A[row][col]; 
  
                // Row is filled, so increase row index and 
                // reset col index 
                if (j == n - 1) 
                { 
                    j = 0; 
                    i++; 
                } 
            } 
        } 
    } 
} 
  
/* Recursive function for finding determinant of matrix. 
   n is current dimension of A[][]. */
template <class T, int N>
__device__ double determinant(T A[N][N], int n) 
{ 
    double D = 0; // Initialize result 
  
    //  Base case : if matrix contains single element 
    if (n == 1) {
        return A[0][0];
	}

	if (n == 2) {
		return (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]); 
	}

	if (n == 3) {
		double _1 = ((double)A[0][0] * (double)A[1][1] * (double)A[2][2]) +
					((double)A[0][1] * (double)A[1][2] * (double)A[2][0]) +
					((double)A[0][2] * (double)A[1][0] * (double)A[2][1]);
		double _2 = ((double)A[0][2] * (double)A[1][1] * (double)A[2][0]) +
					((double)A[0][1] * (double)A[1][0] * (double)A[2][2]) +
					((double)A[0][0] * (double)A[1][2] * (double)A[2][1]);
		return _1 - _2;
	}
  
    T temp[N][N]; // To store cofactors 
  
    double sign = 1;  // To store sign multiplier 
  
     // Iterate for each element of first row 
    for (int f = 0; f < n; f++) 
    { 
        // Getting Cofactor of A[0][f] 
        getCofactor<T,N>(A, temp, 0, f, n); 
        D += (sign) * (double)A[0][f] * determinant<T,N>(temp, n - 1);

        // terms are to be added with alternate sign 
        sign = -sign; 
    } 

    return D; 
} 

template <class T>
__device__ double determinant3(T A[3][3]) {
	double _1 = ((double)A[0][0] * (double)A[1][1] * (double)A[2][2]) +
				((double)A[0][1] * (double)A[1][2] * (double)A[2][0]) +
				((double)A[0][2] * (double)A[1][0] * (double)A[2][1]);
	double _2 = ((double)A[0][2] * (double)A[1][1] * (double)A[2][0]) +
				((double)A[0][1] * (double)A[1][0] * (double)A[2][2]) +
				((double)A[0][0] * (double)A[1][2] * (double)A[2][1]);
	return _1 - _2;
}

template <class T>
__device__ double determinant2(T A[2][2]) {
	return (A[0][0] * A[1][1]) - (A[0][1] * A[1][0]);
}
  
// Function to get adjoint of A[N][N] in adj[N][N].
template <class T, int N>
__device__ void adjoint(T A[N][N], double adj[N][N]) 
{ 
    if (N == 1) 
    { 
        adj[0][0] = 1; 
        return; 
    } 
  
    // temp is used to store cofactors of A[][] 
    double sign = 1;
	T temp[N][N]; 
  
    for (int i=0; i<N; i++) 
    { 
        for (int j=0; j<N; j++) 
        { 
            // Get cofactor of A[i][j] 
            getCofactor<T,N>(A, temp, i, j, N); 
  
            // sign of adj[j][i] positive if sum of row 
            // and column indexes is even. 
            sign = ((i+j)%2==0)? 1: -1; 
  
            // Interchanging rows and columns to get the 
            // transpose of the cofactor matrix 
            adj[j][i] = (sign)*(determinant<T,N>(temp, N-1)); 
        } 
    } 
} 
  
// Function to calculate and store inverse, returns false if 
// matrix is singular
template <class T, int N> 
__device__ bool inverse(T A[N][N], T inverse[N][N]) 
{ 
    // Find determinant of A[][] 
    double det = determinant<T,N>(A, N);
    if (det == 0) 
    { 
        return false; 
    } 
  
    // Find adjoint 
    double adj[N][N]; 
    adjoint<T,N>(A, adj);
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
    for (int i=0; i<N; i++) 
        for (int j=0; j<N; j++) 
            inverse[i][j] = adj[i][j]/det; 
  
    return true; 
}

template <class T> 
__device__ bool inverse3(T A[3][3], T inverse[3][3]) 
{ 
    // Find determinant of A[][] 
    double det = determinant3<T>(A);
    // if (det == 0) 
    // { 
    //     return false; 
    // } 
  
    // Find adjoint 
    // double adj[N][N]; 
    // adjoint<T,N>(A, adj);
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	#pragma unroll 3
    for (int i=0; i<3; i++) 
		#pragma unroll 3
        for (int j=0; j<3; j++) 
            inverse[i][j] = ((A[(j+1)%3][(i+1)%3] * A[(j+2)%3][(i+2)%3]) - (A[(j+1)%3][(i+2)%3] * A[(j+2)%3][(i+1)%3]))/ det;
  
    return true; 
}

template <class T> 
__device__ bool inverse2(T A[2][2], T inverse[2][2]) 
{ 
    // Find determinant of A[][] 
    double det = determinant2<T>(A);
    // if (det == 0) 
    // { 
    //     return false; 
    // } 
  
    // Find adjoint 
    // double adj[N][N]; 
    // adjoint<T,N>(A, adj);
  
    // Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	#pragma unroll 2
    for (int i=0; i<2; i++) 
		#pragma unroll 2
        for (int j=0; j<2; j++) 
            inverse[i][j] = ((A[(j+1)%2][(i+1)%2] * A[(j+2)%2][(i+2)%2]) - (A[(j+1)%2][(i+2)%2] * A[(j+2)%2][(i+1)%2]))/ det;
  
    return true; 
}

template <class T, int N>
__inline__ __device__ void matrix_x_vector(int n, T y[N], T x[N][N], T A[N])
{
  int i, j; // i = row; j = column;

  // Load up A[n][n]
  for (i=0; i<n; i++)
  {
    for (j=0; j<n; j++)
    {
      A[j] += x[j][i] * y[i];
    }

  }
}

template <class T>
__device__ void matrix_x_vector3(T y[3], T x[3][3], T A[3])
{
  int i, j; // i = row; j = column;

  // Load up A[n][n]
  #pragma unroll 3
  for (i=0; i<3; i++)
  {
	#pragma unroll 3
    for (j=0; j<3; j++)
    {
      A[j] += x[j][i] * y[i];
    }

  }
}

template <class T>
__device__ void matrix_x_vector2(T y[2], T x[2][2], T A[2])
{
  int i, j; // i = row; j = column;

  // Load up A[n][n]
  #pragma unroll 2
  for (i=0; i<2; i++)
  {
	#pragma unroll 2
    for (j=0; j<2; j++)
    {
      A[j] += x[j][i] * y[i];
    }

  }
}
