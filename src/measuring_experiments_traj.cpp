#include <nlohmann/json.hpp>
#include <frenet_optimal_trajectory_cuda.hpp>
#include <frenet_optimal_trajectory.hpp>
#include <cubic_spline_planner.hpp>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iomanip>
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

struct Result {
    std::vector<double> x, y;
};

Result path_test(int iterations, FrenetType frenet_type, int path_length, int n_paths)
{

	// Load parameters
    std::ifstream params_file("./cfg/params.json");
    json json_params;
    params_file >> json_params;

	Params<prec_type> params = {
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

    params.dt = params.maxt / (path_length-2);
    double minV = params.target_speed - params.d_t_s*params.n_s_sample;
    double maxV = params.target_speed + params.d_t_s*params.n_s_sample;
    const int v_sample = (maxV-minV+2)/params.d_t_s;
    params.d_road_w = (params.max_road_width_right+params.max_road_width_left)*v_sample / (n_paths);

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
    Result res;
	for(int i = 0; i < iterations; i++)
	{
		FrenetPath<double> path;
		switch (frenet_type)
		{
			case CPU:
				path = planner.frenet_optimal_planning(s0, s_d, c_d, c_d_d, c_d_dd, obs, first_path, last_path, 0);
				break;
			case GPU:
				path = planner_gpu.frenet_optimal_planning(s0, s_d, c_d, c_d_d, c_d_dd, obs, first_path, last_path, 0);
				break;
			default:
				break;
		}

        if (!path.empty){
            int j = 1;

            // Move the actual state to the first point of the path generated
            s0 = path.s[j];
            c_d = path.d[j];
            c_d_d = path.d_d[j];
            c_d_dd = path.d_dd[j];
            s_d = path.s_d[j];

            res.x.push_back(path.x[j]);
            res.y.push_back(path.y[j]);
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

    return res;
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(6) << std::fixed;
    int iteration = 227;
    std::string frnete_type_string;
    switch (FRENET_TYPE) {
        case CPU:
            frnete_type_string = "CPU";
            break;
        case GPU:
            frnete_type_string = "GPU";
            break;
        default:
            frnete_type_string = "UNRECOGNIZED";

    }
    switch (prec_type_t) {
        case double_prec:
            frnete_type_string += "_double";
            break;
        case half_prec:
            frnete_type_string += "_half";
            break;
        case float_prec:
            frnete_type_string += "_float";
            break;
        default:
            frnete_type_string += "_UNRECOGNIZED";
    }


    int path_length = 1024;
    int n_paths = 1024;
    ofstream fex1;
    std::ostringstream fex1Name;
    fex1Name << frnete_type_string << "_traj.csv";
    fex1.open(fex1Name.str());
    fex1 << "PATH LENGTH (" << path_length <<") - N PATHS (" << n_paths << ") - " << frnete_type_string << endl;
    fex1 << "Iteration\tx\ty" << endl;
    Result res = path_test(iteration, FRENET_TYPE, path_length, n_paths);
    for (int i=0; i<res.x.size(); i++){
        fex1 << i << "\t" << res.x[i] << "\t" << res.y[i] << endl;
    }
    fex1.close();

	return 0;
}