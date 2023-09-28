#include <fstream>
#include <chrono>
#include <nlohmann/json.hpp>
#include "matplotlibcpp.h"
using json = nlohmann::json;

#define half_prec 0
#define float_prec 1
#define double_prec 2

#ifndef prec_type_t
#define prec_type_t double_prec
#endif

#if prec_type_t == half_prec
typedef float prec_type;
#elif prec_type_t == float_prec
typedef float prec_type;
#elif prec_type_t == double_prec
typedef double prec_type;
#endif

#define CPU 0
#define GPU 1

#ifndef FRENET_TYPE
    #define FRENET_TYPE CPU
#endif

#if FRENET_TYPE == CPU
	#include "../include/frenet_optimal_trajectory.hpp"
    #define fileName "cpu_test_traj.csv"
#elif FRENET_TYPE == GPU
	#include "../include/frenet_optimal_trajectory_cuda.hpp"
	#define FrenetPlanner FrenetPlannerGPU
#if prec_type_t == double_prec
    #define fileName "gpu_double_test_traj.csv"
#elif prec_type_t == float_prec
    #define fileName "gpu_float_test_traj.csv"
#elif prec_type_t == half_prec
    #define fileName "gpu_half_test_traj.csv"
#endif
#endif

using namespace matplotlibcpp;

void path_test()
{

	// Load parameters
    std::ifstream params_file("./cfg/params.json");
    json json_params;
    params_file >> json_params;

	Params<prec_type>params = {
	json_params["max_speed"],
	json_params["max_accel"],
	json_params["max_curvature"],
	json_params["max_road_width"],
	json_params["d_road_w"],
	json_params["dt"],
	json_params["maxt"],
	json_params["mint"],
	json_params["target_speed"],
	json_params["d_t_s"],
	json_params["n_s_sample"],
	json_params["robot_radius"],
	json_params["max_road_width_left"],
	json_params["max_road_width_right"],
	json_params["safe_distance"],
	json_params["range_path_check"],
	json_params["next_s_borders"],
	json_params["kj"],
	json_params["kt"],
	json_params["kd"],
	json_params["klat"],
	json_params["klon"],
	json_params["check_derivatives"],
	};

	int iteration_limit = json_params["iteration_limit"];
	double to_goal = json_params["distance_to_goal"];

	// Load track
	std::ifstream paths_file("./cfg/track.json");
    json json_paths;
    paths_file >> json_paths;

	vecD wx = json_paths["X"];
	vecD wy = json_paths["Y"];
	vecD wx_inner = json_paths["X_i"];
	vecD wy_inner = json_paths["Y_i"];
	vecD wx_outer = json_paths["X_o"];
	vecD wy_outer = json_paths["Y_o"];

	// Create reference, inner and outer spline
	vecD tx, ty, tyaw, tc;	
	Spline2D spline_ref = calc_spline_course(wx, wy, tx, ty, tyaw, tc, 0.1);
	Spline2D spline_inner = calc_spline_course(wx_inner, wy_inner, tx, ty, tyaw, tc, 0.1);
	Spline2D spline_outer = calc_spline_course(wx_outer, wy_outer, tx, ty, tyaw, tc, 0.1);

	// Init Frenet planner class
	FrenetPlanner<prec_type> planner = FrenetPlanner<prec_type>(params, spline_ref, spline_inner, spline_outer);

	double s_d = 80.0, c_d = 0.0, c_d_d = 0.0, c_d_dd = 0.0, s0 = 5.0;

	//----------------Simulate other agents

	std::vector<obstacle> obs;
	double x_temp,y_temp;
//	spline_ref.calc_position(&x_temp, &y_temp, 3500);
//	obstacle ob_a = {
//		.x = x_temp,
//		.y = y_temp,
//		.radius = 3.0,
//		.s = 3500,
//		.d = 0
//	};
//	obs.push_back(ob_a);

	spline_ref.calc_position(&x_temp, &y_temp, 600);
	obstacle ob_b = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 600,
		.d = 0
	};
	obs.push_back(ob_b);
//
//	spline_ref.calc_position(&x_temp, &y_temp, 1100);
//	obstacle ob_c = {
//		.x = x_temp,
//		.y = y_temp,
//		.radius = 3.0,
//		.s = 1100,
//		.d = 0
//	};
//	obs.push_back(ob_c);

//	spline_ref.calc_position(&x_temp, &y_temp, 2500);
//	obstacle ob_d = {
//		.x = x_temp,
//		.y = y_temp,
//		.radius = 3.0,
//		.s = 2500,
//		.d = 0
//	};
//	obs.push_back(ob_d);
//
//	spline_ref.calc_position(&x_temp, &y_temp, 2000);
//	obstacle ob_e = {
//		.x = x_temp,
//		.y = y_temp,
//		.radius = 3.0,
//		.s = 2000,
//		.d = 0
//	};
//	obs.push_back(ob_e);
//
//	spline_ref.calc_position(&x_temp, &y_temp, 3000);
//	obstacle ob_f = {
//		.x = x_temp,
//		.y = y_temp,
//		.radius = 3.0,
//		.s = 3000,
//		.d = 0
//	};
//	obs.push_back(ob_f);

	//----------------------

	std::vector<FrenetPath<double>> log_paths;
	std::vector<FrenetPath<double>> log_first_paths;
	std::vector<FrenetPath<double>> log_last_paths;
    FrenetPath<double> first_path, last_path;

	std::vector<std::vector<double>> obs_x, obs_y;

	// Run the simulation
	for(int i = 0; i < iteration_limit; i++)
	{

		auto t_before = std::chrono::high_resolution_clock::now();
		FrenetPath<double> path = planner.frenet_optimal_planning(s0, s_d, c_d, c_d_d, c_d_dd, obs, first_path, last_path, 0);
        std::chrono::duration<double> time_it= std::chrono::duration_cast<std::chrono::duration<double>>((std::chrono::high_resolution_clock::now()) - t_before);
        std::cout << "-------------------------> Time Iteration -> " << time_it.count()*1000 << std::endl;

		int j = 1;

		// Move the actual state to the first point of the path generated
		s0 = path.s[j];
		c_d = path.d[j];
		c_d_d = path.d_d[j];
		c_d_dd = path.d_dd[j];
		s_d = path.s_d[j];

		// Move the other agents along the reference path
		std::vector<double> temp_x, temp_y;
		for(int k = 0; k<obs.size(); ++k)
		{
//			obs[k].s = obs[k].s + 2;
//
//			if(obs[k].s > spline_ref.get_s_last())
//			{
//				obs[k].s = 0;
//			}

			spline_ref.calc_position(&(obs[k].x), &(obs[k].y), obs[k].s);

			temp_x.push_back(obs[k].x);
			temp_y.push_back(obs[k].y);
		}
		obs_x.push_back(temp_x);
		obs_y.push_back(temp_y);

		// Check to end the simulation
		double distance_to_goal = sqrt(pow(path.x[j] - wx[0], 2) + pow(path.y[j] - wy[0], 2));
		if(distance_to_goal <= to_goal)
		{
			std::cout << "Distance to goal: "<< distance_to_goal << " - Goal reached!"<< std::endl;
			std::cout << "Iterations: " << i << std::endl;
			break;
		}

		std::cout << "Iteration: " << i << std::endl;
		std::cout << "Printing the path params :" << std::endl;
		std::cout << std::to_string(s0) << " \n" << std::to_string(c_d) 
		<< " \n" << std::to_string(c_d_d) << " \n" << std::to_string(c_d_dd) << " \n" << std::to_string(s_d) << std::endl;

		//FrenetPathPlot path_plot(path), first_path_plot(*first_path), last_path_plot(*last_path);
		log_paths.push_back(path);
		log_first_paths.push_back(first_path);
		log_last_paths.push_back(last_path);

	}

	//Plot the simulation
    ofstream fex1, fex2;
     fex1.open(fileName);
    fex1 << fileName << endl;
    fex1 << "Iteration\tx\ty\tob1_x\tob1_y" << endl;
    for (int i=0; i<log_paths[117].x.size(); i++) {
        fex1 << i << "\t" << log_paths[117].x[i] << "\t" << log_paths[117].y[i] << "\t" << obs_x[117][0] << "\t" << obs_y[117][0] << std::endl;
    }
//	 for(int i=0; i<log_paths.size(); ++i)
//	 {
//
//	 	clf();
//
//	 	subplot(1,2,1);
//	 	plot(wx, wy, "b");
//	 	plot(wx_inner, wy_inner, "k");
//	 	plot(wx_outer, wy_outer, "k");
//	 	double distance_to_goal = sqrt(pow(log_paths[i].x[1] - wx[0], 2) + pow(log_paths[i].y[1] - wy[0], 2));
//	 	plot(log_paths[i].x, log_paths[i].y, ".g");
//	 	plot(log_first_paths[i].x, log_first_paths[i].y, "-r");
//	 	plot(log_last_paths[i].x, log_last_paths[i].y, "-r");
//	 	plot(obs_x[i], obs_y[i], ".r");
//
//	 	subplot(1,2,2);
//	 	plot(wx, wy, "b");
//	 	plot(wx_inner, wy_inner, "k");
//	 	plot(wx_outer, wy_outer, "k");
//	 	plot(log_paths[i].x, log_paths[i].y, ".g");
//	 	plot(log_first_paths[i].x, log_first_paths[i].y, "-r");
//	 	plot(log_last_paths[i].x, log_last_paths[i].y, "-r");
//	 	plot(obs_x[i], obs_y[i], ".r");
//
//
//
//
//	 	xlabel("Iteration: "+std::to_string(i));
//
//	 	xlim(log_paths[i].x[0]-100, log_paths[i].x[0]+100);
//	 	ylim(log_paths[i].y[0]-100, log_paths[i].y[0]+100);
//
//	 	// xlim(obs_x[i][1]-10, obs_x[i][1]+10);
//	 	// ylim(obs_y[i][1]-10, obs_y[i][1]+10);
//
//	 	pause(0.0001);
//
//	 }

}

int main()
{	
	path_test();
	return 0;
}


