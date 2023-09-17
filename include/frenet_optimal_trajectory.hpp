#ifndef FRENET_PLANNER_HPP
#define FRENET_PLANNER_HPP

#include <algorithm>
#include <cfloat>
#include <cubic_spline_planner.hpp>
#include <cpp_common_structs/structures.hpp>


template <class T>
struct Params{

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
    T next_s_borders;

    T kj; //jerk cost
    T kt; //convergence time cost
    T kd; //lateral deviation cost
    T klat;
    T klon;

	bool check_derivatives;

};

template <class T>
class FrenetPlanner
{
	public:
		FrenetPlanner(){};
		FrenetPlanner(Params<T> params):params_(params){};
		FrenetPlanner(Params<T> params, Spline2D reference_path, Spline2D i_border, Spline2D o_border)
		:params_(params),
		reference_path_(reference_path),
		i_border_(i_border),
		o_border_(o_border){};

		~FrenetPlanner(){};

		FrenetPath<double> frenet_optimal_planning(double s0, double s_d, double c_d, double c_d_d, double c_d_dd,
					vector<obstacle> obstacles, FrenetPath<double> &first, FrenetPath<double> &last, int overtake_strategy);

#ifdef MEASURE_TASK_TIME
        double time_calc, time_check;
#endif

	private:
		Params<T> params_;
		Spline2D reference_path_, i_border_, o_border_;

        vector<FrenetPath<T>> calc_frenet_paths(T s_d, T c_d, T c_d_d, T c_d_dd, T s0);
		vector<FrenetPath<T>> calc_global_paths(vector<FrenetPath<T>> fplist);

		bool check_collision_path(FrenetPath<T>, vector<obstacle> );
		vector<FrenetPath<T>> check_path(vector<FrenetPath<T>> fplist, vector<obstacle> obs);
		bool check_derivatives(T s_d, T s_dd, T c);


};

#endif
