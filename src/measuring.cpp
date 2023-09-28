#include <nlohmann/json.hpp>
#include <frenet_optimal_trajectory_cuda.hpp>
#include <frenet_optimal_trajectory.hpp>
#include <cubic_spline_planner.hpp>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
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

enum FrenetType {CPU, GPU};

void path_test(int iterations, FrenetType frenet_type)
{

	// Load parameters
    std::ifstream params_file("./cfg/params.json");
    json json_params;
    params_file >> json_params;

	Params<prec_type>params = {
    (prec_type)json_params["max_speed"],
    (prec_type)json_params["max_accel"],
    (prec_type)json_params["max_curvature"],
    (prec_type)json_params["max_road_width"],
    (prec_type)json_params["d_road_w"],
    (prec_type)json_params["dt"],
    (prec_type)json_params["maxt"],
    (prec_type)json_params["mint"],
    (prec_type)json_params["target_speed"],
    (prec_type)json_params["d_t_s"],
    (prec_type)json_params["n_s_sample"],
    (prec_type)json_params["robot_radius"],
    (prec_type)json_params["max_road_width_left"],
    (prec_type)json_params["max_road_width_right"],
    (prec_type)json_params["safe_distance"],
    (prec_type)json_params["range_path_check"],
    (prec_type)json_params["next_s_borders"],
    (prec_type)json_params["kj"],
    (prec_type)json_params["kt"],
    (prec_type)json_params["kd"],
    (prec_type)json_params["klat"],
    (prec_type)json_params["klon"],
	json_params["check_derivatives"],
	};

	//int iteration_limit = json_params["iteration_limit"];
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

	double s_d = 80.0, c_d = 0.0, c_d_d = 0.0, c_d_dd = 0.0, s0 = 5.0;

	//----------------Simulate other agents

	std::vector<obstacle> obs;
	double x_temp,y_temp;
	spline_ref.calc_position(&x_temp, &y_temp, 3500);
	obstacle ob_a = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 3500,
		.d = 0
	};
	obs.push_back(ob_a);

	spline_ref.calc_position(&x_temp, &y_temp, 600);
	obstacle ob_b = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 600,
		.d = 0
	};
	obs.push_back(ob_b);

	spline_ref.calc_position(&x_temp, &y_temp, 1100);
	obstacle ob_c = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 1100,
		.d = 0
	};
	obs.push_back(ob_c);

	spline_ref.calc_position(&x_temp, &y_temp, 2500);
	obstacle ob_d = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 2500,
		.d = 0
	};
	obs.push_back(ob_d);

	spline_ref.calc_position(&x_temp, &y_temp, 2000);
	obstacle ob_e = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 2000,
		.d = 0
	};
	obs.push_back(ob_e);

	spline_ref.calc_position(&x_temp, &y_temp, 3000);
	obstacle ob_f = {
		.x = x_temp,
		.y = y_temp,
		.radius = 3.0,
		.s = 3000,
		.d = 0
	};
	obs.push_back(ob_f);

	// Init Frenet planner class
    FrenetPlanner<prec_type> planner(params, spline_ref, spline_inner, spline_outer);
	FrenetPlannerGPU<prec_type> planner_gpu(params, spline_ref, spline_inner, spline_outer);

	// Run the simulation
    FrenetPath<double> first_path, last_path;
	std::vector<double> measures;
	for(int i = 0; i < iterations; i++)
	{

		auto t_before = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_it;
		FrenetPath<double> path;
		switch (frenet_type)
		{
			case CPU:
				t_before = std::chrono::high_resolution_clock::now();
				path = planner.frenet_optimal_planning(s0, s_d, c_d, c_d_d, c_d_dd, obs, first_path, last_path, 0);
				time_it= std::chrono::duration_cast<std::chrono::duration<double>>((std::chrono::high_resolution_clock::now()) - t_before);
				break;
			case GPU:
				t_before = std::chrono::high_resolution_clock::now();
				path = planner_gpu.frenet_optimal_planning(s0, s_d, c_d, c_d_d, c_d_dd, obs, first_path, last_path, 0);
				time_it= std::chrono::duration_cast<std::chrono::duration<double>>((std::chrono::high_resolution_clock::now()) - t_before);
				break;
			default:
				break;
		}
        std::cout << "Time Iteration -> " << time_it.count()*1000 << std::endl;
		double time_d = std::chrono::duration<double>(time_it).count();
		measures.push_back(time_d*1000);

        if (!path.empty){
            int j = 1;

            // Move the actual state to the first point of the path generated
            s0 = path.s[j];
            c_d = path.d[j];
            c_d_d = path.d_d[j];
            c_d_dd = path.d_dd[j];
            s_d = path.s_d[j];
        }

		// Move the other agents along the reference path
		for(int k = 0; k<obs.size(); ++k)
		{
			obs[k].s = obs[k].s + 2;

			if(obs[k].s > spline_ref.get_s_last())
			{
				obs[k].s = 0;
			}

			spline_ref.calc_position(&(obs[k].x), &(obs[k].y), obs[k].s);

		}

	}

	std::cout << std::endl;
	std::cout << "Performed " << iterations << " iterations" << std::endl;
	double avg = 0;
	double max = 0;
	double min = std::numeric_limits<double>::max();
	for (double el : measures){
		avg += el;
		if (min > el)
			min = el;
		if (max < el)
			max = el;
	}
	avg = avg / measures.size();
	// auto min = std::min_element(measures.begin(), measures.end());
	// auto max = std::max_element(measures.begin(), measures.end());
	// auto mean = std::accumulate(measures.begin(), measures.end(), 0) / (double)measures.size();
	std::cout << "Time --> Avg: " << avg << " Max: " << max << " Min: " << min << std::endl;

}

int main(int argc, char *argv[])
{	
    if (argc < 3){
        std::cout << "Specificare TIPO(CPU/GPU) e NUMERO_ITERAZIONI"<<std::endl;
        return 1;
    }
    int iteration = atoi(argv[2]);
    FrenetType frenet_type;
    if (strcmp(argv[1], "GPU") == 0){
        frenet_type = GPU;
    } else if (strcmp(argv[1], "CPU") == 0){
        frenet_type = CPU;
    } else {
        std::cout << "Tipo errato!" << std::endl;
        return 1;
    } 

	path_test(iteration, frenet_type);
	return 0;
}