#include <nlohmann/json.hpp>
#include <frenet_optimal_trajectory.hpp>
#include <frenet_optimal_trajectory_cuda.hpp>
#include <cubic_spline_planner.hpp>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <iomanip>

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

using json = nlohmann::json;

enum FrenetType {CPU, GPU};

struct Result {
    double avg, median, min, max, std;
#ifdef MEASURE_TASK_TIME
    double avg_calc, median_calc, min_calc, max_calc, std_calc;
    double avg_check, median_check, min_check, max_check, std_check;
#if FRENET_TYPE == GPU
    double avg_calc_mem, median_calc_mem, min_calc_mem, max_calc_mem, std_calc_mem;
    double avg_check_mem, median_check_mem, min_check_mem, max_check_mem, std_check_mem;
#endif
#endif
};

double calcDeviation(const vector<double>& v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

void calcResult(std::vector<double> &measures, double &avg, double &median, double &max, double &min, double &st_dev) {
    avg = 0;
    max = 0;
    min = std::numeric_limits<double>::max();
    for (double el : measures){
        avg += el;
        if (min > el)
            min = el;
        if (max < el)
            max = el;
    }
    avg = avg / measures.size();
    sort(measures.begin(),measures.end());
    median = measures[measures.size()/2];
    st_dev = calcDeviation(measures, avg);
}

Result path_test(int iterations, FrenetType frenet_type, int path_length, int n_paths)
{

	// Load parameters
    std::ifstream params_file("./cfg/params.json");
    json json_params;
    params_file >> json_params;

	Params<prec_type> params = {
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
	std::vector<double> measures;
#ifdef MEASURE_TASK_TIME
    std::vector<double> calc_times, check_times;
#if FRENET_TYPE == GPU
    std::vector<double> calc_times_mem, check_times_mem;
#endif
#endif
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
#ifdef MEASURE_TASK_TIME
                calc_times.push_back(planner.time_calc);
                check_times.push_back(planner.time_check);
#endif
				break;
			case GPU:
				t_before = std::chrono::high_resolution_clock::now();
				path = planner_gpu.frenet_optimal_planning(s0, s_d, c_d, c_d_d, c_d_dd, obs, first_path, last_path, 0);
				time_it= std::chrono::duration_cast<std::chrono::duration<double>>((std::chrono::high_resolution_clock::now()) - t_before);
#ifdef MEASURE_TASK_TIME
                calc_times.push_back(planner_gpu.time_calc);
                check_times.push_back(planner_gpu.time_check);
                //calc_times_mem.push_back(planner_gpu.time_calc_mem);
                //check_times_mem.push_back(planner_gpu.time_check_mem);
#endif
				break;
			default:
				break;
		}
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

    Result res;
    calcResult(measures, res.avg, res.median, res.max, res.min, res.std);
#ifdef MEASURE_TASK_TIME
    calcResult(calc_times, res.avg_calc, res.median_calc, res.max_calc, res.min_calc, res.std_calc);
    calcResult(check_times, res.avg_check, res.median_check, res.max_check, res.min_check, res.std_check);
#if FRENET_TYPE == GPU
    //calcResult(calc_times_mem, res.avg_calc_mem, res.median_calc_mem, res.max_calc_mem, res.min_calc_mem, res.std_calc_mem);
    //calcResult(check_times_mem, res.avg_check_mem, res.median_check_mem, res.max_check_mem, res.min_check_mem, res.std_check_mem);
#endif
#endif

    return res;
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(6) << std::fixed;
    int iteration = 100;
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
    //Ex 1
    int path_length = 1024;
    ofstream fex1;
    std::ostringstream fex1Name;
#ifdef MEASURE_TASK_TIME
    fex1Name << "tasks_";
#endif
    fex1Name << frnete_type_string << "_Ex1.csv";
    fex1.open(fex1Name.str());
    fex1 << "SAME PATH LENGTH (" << path_length <<") - VARYING N PATHS" << " Performed " << iteration << " iterations" << endl;
    fex1 << "Path Length\tN Paths";
#ifdef MEASURE_TASK_TIME
    fex1 << "\tAvg_calc_"<<frnete_type_string<<"\tMedian_calc_"<<frnete_type_string<<"\tMax_calc_"<<frnete_type_string<<"\tMin_calc_"<<frnete_type_string<<"\tStd_calc_"<<frnete_type_string;
    fex1 << "\tAvg_check_"<<frnete_type_string<<"\tMedian_check_"<<frnete_type_string<<"\tMax_check_"<<frnete_type_string<<"\tMin_check_"<<frnete_type_string<<"\tStd_check_"<<frnete_type_string << endl;
#else
    fex1 << "\tAvg_"<<frnete_type_string<<"\tMedian_"<<frnete_type_string<<"\tMax_"<<frnete_type_string<<"\tMin_"<<frnete_type_string<<"\tStd_"<<frnete_type_string << endl;
#endif
    for (int n_paths = 128; n_paths <= 2048; n_paths += 64) {
        Result res = path_test(iteration, FRENET_TYPE, path_length, n_paths);
#ifdef MEASURE_TASK_TIME
        fex1 << path_length << "\t" << n_paths << "\t" << res.avg_calc << "\t" << res.median_calc << "\t" << res.max_calc << "\t" << res.min_calc << "\t" << res.std_calc;
        fex1 << "\t" << res.avg_check << "\t" << res.median_check << "\t" << res.max_check << "\t" << res.min_check << "\t" << res.std_check << std::endl;
        std::cout << path_length << "\t" << n_paths << "\t" << res.avg_calc << "\t" << res.median_calc << "\t" << res.max_calc << "\t" << res.min_calc << "\t" << res.std_calc;
        std::cout << "\t" << res.avg_check << "\t" << res.median_check << "\t" << res.max_check << "\t" << res.min_check << "\t" << res.std_check << std::endl;
#else
        fex1 << path_length << "\t" << n_paths << "\t" << res.avg << "\t" << res.median << "\t" << res.max << "\t" << res.min << "\t" << res.std << std::endl;
        std::cout << path_length << "\t" << n_paths << "\t" << res.avg << "\t" << res.median << "\t" << res.max << "\t" << res.min << "\t" << res.std << std::endl;
#endif
    }
    fex1.close();

    std::cout << "----------------------------------" << endl;

    //Ex 2
    int n_paths = 1024;
    ofstream fex2;
    std::ostringstream fex2Name;
#ifdef MEASURE_TASK_TIME
    fex2Name << "tasks_";
#endif
    fex2Name << frnete_type_string << "_Ex2.csv";
    fex2.open(fex2Name.str());
    fex2 << "SAME N PATHS (" << n_paths <<") - VARYING PATH LENGTH" << " Performed " << iteration << " iterations" << endl;
    fex2 << "Path Length\tN Paths";
#ifdef MEASURE_TASK_TIME
    fex2 << "\tAvg_calc_"<<frnete_type_string<<"\tMedian_calc_"<<frnete_type_string<<"\tMax_calc_"<<frnete_type_string<<"\tMin_calc_"<<frnete_type_string<<"\tStd_calc_"<<frnete_type_string;
    fex2 << "\tAvg_check_"<<frnete_type_string<<"\tMedian_check_"<<frnete_type_string<<"\tMax_check_"<<frnete_type_string<<"\tMin_check_"<<frnete_type_string<<"\tStd_check_"<<frnete_type_string << endl;
#else
    fex1 << "\tAvg_"<<frnete_type_string<<"\tMedian_"<<frnete_type_string<<"\tMax_"<<frnete_type_string<<"\tMin_"<<frnete_type_string<<"\tStd_"<<frnete_type_string << endl;
#endif
    for (int  path_length = 64; path_length <= 1024; path_length += 32) {
        Result res = path_test(iteration, FRENET_TYPE, path_length, n_paths);
#ifdef MEASURE_TASK_TIME
        fex2 << path_length << "\t" << n_paths << "\t" << res.avg_calc << "\t" << res.median_calc << "\t" << res.max_calc << "\t" << res.min_calc << "\t" << res.std_calc;
        fex2 << "\t" << res.avg_check << "\t" << res.median_check << "\t" << res.max_check << "\t" << res.min_check << "\t" << res.std_check << std::endl;
        std::cout << path_length << "\t" << n_paths << "\t" << res.avg_calc << "\t" << res.median_calc << "\t" << res.max_calc << "\t" << res.min_calc << "\t" << res.std_calc;
        std::cout << "\t" << res.avg_check << "\t" << res.median_check << "\t" << res.max_check << "\t" << res.min_check << "\t" << res.std_check << std::endl;
#else
        fex1 << path_length << "\t" << n_paths << "\t" << res.avg << "\t" << res.median << "\t" << res.max << "\t" << res.min << "\t" << res.std << std::endl;
        std::cout << path_length << "\t" << n_paths << "\t" << res.avg << "\t" << res.median << "\t" << res.max << "\t" << res.min << "\t" << res.std << std::endl;
#endif
    }
    fex2.close();

	return 0;
}