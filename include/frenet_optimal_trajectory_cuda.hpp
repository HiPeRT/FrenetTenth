#ifndef FRENET_PLANNER_CUDA_HPP
#define FRENET_PLANNER_CUDA_HPP

#include <algorithm>
#include <cfloat>
#include <cubic_spline_planner.hpp>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/frenet_optimal_trajectory.hpp"


inline int floor(__half x) {
    return floor((float)x);
}

template <class T>
struct ParamsGPU{

	ParamsGPU() = default;

	ParamsGPU(const struct Params<T> &p) {
		this->max_speed = p.max_speed;		
		this->max_accel = p.max_accel;
		this->max_curvature = p.max_curvature;
		this->max_road_width = p.max_road_width;
		this->d_road_w = p.d_road_w;
		this->dt = p.dt;
		this->maxt = p.maxt;
		this->mint = p.mint;
		this->target_speed = p.target_speed;
		this->d_t_s = p.d_t_s;
		this->n_s_sample = p.n_s_sample;
		this->robot_radius = p.robot_radius;
		this->max_road_width_left = p.max_road_width_left;
		this->max_road_width_right = p.max_road_width_right;

		this->safe_distance = p.safe_distance;
		this->range_path_check = p.range_path_check;
		this->next_s_borders = p.next_s_borders;

		this->kj = p.kj;
		this->kt = p.kt;
		this->kd = p.kd;
		this->klat = p.klat;
		this->klon = p.klon;
	};

	T max_speed; // maximum speed [m/s]
	T max_accel;  // maximum acceleration [m/ss]
	T max_curvature;  // maximum curvature [1/m]
	T max_road_width;  // maximum road width [m]
	T d_road_w;  // road width sampling length [m]
	T dt;  // time tick [s]
	T maxt;  // max prediction time [m]
	T mint;  // min prediction time [m]
	T target_speed;  // target speed [m/s]
	T d_t_s;   // target speed sampling length [m/s]
	T n_s_sample;// sampling number of target speed
	T robot_radius;  // robot radius [m]
	T max_road_width_left;
	T max_road_width_right;

	T safe_distance; //distances greater than this value are considered collision free
	T range_path_check; //path portion which should be checked for collision. If 1 means all the path
	int next_s_borders;

	T kj; //jerk cost
	T kt; //convergence time cost
	T kd; //lateral deviation cost
	T klat;
	T klon;
};


template <class T>
struct FrenetPathPoint{
    T t, d, d_d, d_dd, d_ddd, s, s_d, s_dd, s_ddd;
    T x, y, yaw, ds, c;
    T cd, cv, cf;
    int obstacle;
    int length;
    T Ti;
    T cost_;
};

template <class T>
class FrenetPlannerGPU
{
	public:
		FrenetPlannerGPU(){};
		//FrenetPlanner(Params params):params_(params){};
		FrenetPlannerGPU(Params<T> params, Spline2D reference_path, Spline2D i_border, Spline2D o_border);

		~FrenetPlannerGPU();

		FrenetPath<double> frenet_optimal_planning(double s0, double s_d, double c_d, double c_d_d, double c_d_dd, std::vector<obstacle> obstacles, FrenetPath<double> &first, FrenetPath<double> &last, int overtake_strategy);

#ifdef MEASURE_TASK_TIME
        double time_calc, time_check;
        double time_calc_mem, time_check_mem;
#endif

private:
		ParamsGPU<T> params_;
		Spline2D reference_path_, i_border_, o_border_;
		double *spline_x_p, *spline_y_p, *spline_ax_p, *spline_ay_p, *spline_bx_p, *spline_by_p, *spline_cx_p, *spline_cy_p, *spline_dx_p, *spline_dy_p;
		FrenetPathPoint<T> *fpp_array_h, *fpp_array_d;
		int N_POINTS, N_PATHS, TOTAL_POINTS, POINTS_TOTAL_SIZE, OB_MAX_SIZE, OB_MAX_N, spline_size, THREADSX, THREADSY, THREADSZ, BLOCKSX, BLOCKSY, BLOCKSZ;
		T last_s_res;
		T minV, maxV;
		dim3 block, thread;
		obstacle *ob_array_d, *ob_array_h;
		cudaStream_t stream, streamObstacles;
        cudaEvent_t obstacleCopyComplete;

#ifdef MEASURE_TASK_TIME
        cudaEvent_t calc_start, calc_end, check_start, check_end;
#endif

		cudaError_t err;

		int best_path_index;

		cublasHandle_t handle;
    	cublasStatus_t stat;

		void calc_frenet_paths(double, double, double, double, double, FrenetPathPoint<T> **, int *, int *, int);
		void init();


};


#endif
