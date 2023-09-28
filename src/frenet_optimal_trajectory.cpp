#include "../include/frenet_optimal_trajectory.hpp"
#include <chrono>
#include <numeric>
#include <polynomials.hpp>


template class FrenetPlanner<double>;
template class FrenetPlanner<float>;
#if defined(__x86_64__) || defined(_M_X64)
typedef float __fp16;
#else
template class FrenetPlanner<__fp16>;
#endif


template <class T>
T dist(T x1, T y1, T x2, T y2)
{
	return sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2));
}

// Generates Frenet Path combining longitudinal and lateral movements
template <class T>
std::vector<FrenetPath<T>> FrenetPlanner<T>::calc_frenet_paths(T s_d, T c_d, T c_d_d, T c_d_dd, T s0)
{
	std::vector<FrenetPath<T>> frenet_paths;
    
	// For each road width step
	for(T di = -params_.max_road_width_right; di < params_.max_road_width_left; di += params_.d_road_w) //di <= params_.max_road_width
	{
		// For each possible longitudinal length
		for(T Ti = params_.mint; Ti <= params_.maxt; Ti += params_.dt)  // Just one iteration means same length paths
		{
			FrenetPath<T> fp;
            
			quintic<T> lat_qp(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti);
            
			//lateral movement
			for(T t = 0.0; t <= Ti + params_.dt; t += params_.dt)  // linspace
			{
				fp.t.push_back(t);
				fp.d.push_back(lat_qp.calc_point(t));
				fp.d_d.push_back(lat_qp.calc_first_derivative(t));
				fp.d_dd.push_back(lat_qp.calc_second_derivative(t));
				fp.d_ddd.push_back(lat_qp.calc_third_derivative(t));
			} 
			T Jp = std::inner_product(fp.d_ddd.begin(), fp.d_ddd.end(), fp.d_ddd.begin(), 0);
			T minV = params_.target_speed - params_.d_t_s*params_.n_s_sample;
			T maxV = params_.target_speed + params_.d_t_s*params_.n_s_sample;

			//longitudinal movement
	    	for(T tv = minV; tv <= maxV + params_.d_t_s; tv += params_.d_t_s)
			{
				FrenetPath<T> tfp = fp;
				quartic<T> lon_qp(s0, s_d, 0.0, tv, 0.0, Ti);

				for(auto const& t : fp.t) 
				{
					tfp.s.push_back(lon_qp.calc_point(t));
					tfp.s_d.push_back(lon_qp.calc_first_derivative(t));
					tfp.s_dd.push_back(lon_qp.calc_second_derivative(t));
					tfp.s_ddd.push_back(lon_qp.calc_third_derivative(t));
				}

				T Js = std::inner_product(tfp.s_ddd.begin(), tfp.s_ddd.end(), tfp.s_ddd.begin(), 0);

				T ds = pow((params_.target_speed - tfp.s_d.back()), 2);
                
				tfp.cd = params_.kj*Jp + params_.kt*Ti + params_.kd*tfp.d.back()*tfp.d.back(); //lateral generation cost
				tfp.cv = params_.kj*Js + params_.kt*Ti + params_.kd*ds; //longitudinal generation cost
				tfp.cf = params_.klat*tfp.cd + params_.klon*tfp.cv; //combined cost functional

				frenet_paths.push_back(tfp);
			}
		}
		
	}
	return frenet_paths;
}

// From Frenet frame to Cartesian
template <class T>
std::vector<FrenetPath<T>> FrenetPlanner<T>::calc_global_paths(std::vector<FrenetPath<T>> fplist)
{
	for(auto& fp : fplist)
	{
		// x,y
		for(int i = 0; i < fp.s.size(); i++)
		{
			double ix_, iy_;
			reference_path_.calc_position(&ix_, &iy_, fp.s[i]);
            T ix = (T)ix_;
            T iy = (T)iy_;

			if(ix == NONE)
				break;
			T iyaw = reference_path_.calc_yaw(fp.s[i]);
			T di = fp.d[i];

			T fx = ix - di*sin(iyaw);
			T fy = iy + di*cos(iyaw);
            
			fp.x.push_back(fx);
			fp.y.push_back(fy);
		}

		// Yaw and distance between s
		for(int i = 0; i < fp.x.size() - 1; i++)
		{
			T dx = fp.x[i + 1] - fp.x[i];
			T dy = fp.y[i + 1] - fp.y[i];

			fp.yaw.push_back(atan2(dy, dx));
			fp.ds.push_back(sqrt(dx*dx + dy*dy));
		}

		fp.yaw.push_back(fp.yaw[-1]);
		fp.ds.push_back(fp.ds[-1]);

		// Curvature
		for(int i = 0; i < fp.yaw.size() - 1; i++)
			fp.c.push_back((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i]);
		
	}

	return fplist;
}

template <class T>
bool FrenetPlanner<T>::check_derivatives(T s_d, T s_dd, T c)
{
	return (abs(s_d) > params_.max_speed || abs(s_dd) > params_.max_accel || abs(c) > params_.max_curvature);
}

template <class T>
bool FrenetPlanner<T>::check_collision_path(FrenetPath<T> fp, std::vector<obstacle> obs)
{
    for (auto const& ob : obs)
    {
        for (int j=0; j<fp.x.size()*params_.range_path_check; j++)
        {
            T x = fp.x[j];
            T y = fp.y[j];
            T ob_x = ob.x;
            T ob_y = ob.y;
            T radius = ob.radius;
			T di = dist(x, y, ob_x, ob_y) - radius;

			if (di < params_.safe_distance)
			{
				return 1;
			}
			if(params_.check_derivatives && check_derivatives(fp.s_d[j], fp.s_dd[j], fp.c[j]))
			{
				return 1;
			}

        }
    }
    return 0;
}

template <class T>
std::vector<FrenetPath<T>> FrenetPlanner<T>::check_path(std::vector<FrenetPath<T>> fplist, std::vector<obstacle> obs)
{
	std::vector<FrenetPath<T>> fplist_final;
	for(int i = 0; i < fplist.size(); i++)
	{
		if(check_collision_path(fplist[i], obs)==0)
		{
			fplist_final.push_back(fplist[i]);
		}

	}
	return fplist_final;
}

template <class T>
FrenetPath<double> get_frenet_paths_vector(FrenetPath<T> path){
    std::vector<FrenetPath<T>> array;
    array.push_back(path);
    return get_frenet_paths_vector(array)[0];
}

template <class T>
std::vector<FrenetPath<double>> get_frenet_paths_vector(std::vector<FrenetPath<T>> array){
    std::vector<FrenetPath<double>> fp_vector;
    fp_vector.reserve(array.size());
    for (int i=0; i<array.size(); i++){
        FrenetPath<T> path = array[i];
        FrenetPath<double> fp;
        for (int j=0; j<(int)path.x.size(); j++){
            fp.t.push_back(path.t[j]);
            fp.d.push_back(path.d[j]);
            fp.d_d.push_back(path.d_d[j]);
            fp.d_dd.push_back(path.d_dd[j]);
            fp.d_ddd.push_back(path.d_ddd[j]);
            fp.s.push_back(path.s[j]);
            fp.s_d.push_back(path.s_d[j]);
            fp.s_dd.push_back(path.s_dd[j]);
            fp.s_ddd.push_back(path.s_ddd[j]);
            fp.x.push_back(path.x[j]);
            fp.y.push_back(path.y[j]);
            fp.yaw.push_back(path.yaw[j]);
            fp.ds.push_back(path.ds[j]);
            fp.c.push_back(path.c[j]);
        }

        fp.cd = path.cd;
        fp.cv = path.cv;
        fp.cf = path.cf;

        fp_vector.push_back(fp);
    }

    return fp_vector;
}

template <class T>
FrenetPath<double> FrenetPlanner<T>::frenet_optimal_planning(double s0, double s_d, double c_d, double c_d_d, double c_d_dd, std::vector<obstacle> obstacles, FrenetPath<double> &first, FrenetPath<double> &last, int overtake_strategy)
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

#ifdef MEASURE_TASK_TIME
    auto t_before_calc = std::chrono::high_resolution_clock::now();
#endif
    std::vector<FrenetPath<T>> fplist = calc_frenet_paths((T)s_d, (T)c_d, (T)c_d_d, (T)c_d_dd, (T)s0);
    fplist = calc_global_paths(fplist);
#ifdef MEASURE_TASK_TIME
    auto t_after_calc = std::chrono::high_resolution_clock::now();
#endif

	first = get_frenet_paths_vector(fplist[0]);
  	last = get_frenet_paths_vector(fplist[fplist.size()-1]);

	obstacle ob_temp;

	//Add the track borders as obstacles
	double x_i, y_i, x_o, y_o;
	double s_i = s0;
	double s_o = s0;
	for(int i=0; i<params_.next_s_borders; ++i)
	{
		s_i += i;
		s_o += i;
		if(s_i > i_border_.get_s_last())
			s_i = 0;

		if(s_o > o_border_.get_s_last())
			s_o = 0;

		i_border_.calc_position(&x_i, &y_i, s_i);
		o_border_.calc_position(&x_o, &y_o, s_o);

		ob_temp = {
		.x = x_i,
		.y = y_i,
		.radius = 1.0,
		.s = s_i,
		.d = 0};

		obstacles.push_back(ob_temp);

		ob_temp = {
		.x = x_o,
		.y = y_o,
		.radius = 1.0,
		.s = s_o,
		.d = 0};

		obstacles.push_back(ob_temp);
	}

#ifdef MEASURE_TASK_TIME
    auto t_before_check = std::chrono::high_resolution_clock::now();
#endif
    fplist = check_path(fplist, obstacles);
#ifdef MEASURE_TASK_TIME
    auto t_after_check = std::chrono::high_resolution_clock::now();
#endif
    
	double min_cost = FLT_MAX;

	FrenetPath<double> bestpath;
	bestpath.empty = true;
	int best_index = -1;
	int index = 0;
	for(auto const& fp : fplist)
	{
		if(min_cost >= fp.cf)
		{
			min_cost = fp.cf;
			best_index = index;
		}
		index++;
	}

    if (best_index != -1) {
        bestpath = get_frenet_paths_vector(fplist[best_index]);
        bestpath.empty = false;
    }

#ifdef MEASURE_TASK_TIME
    time_calc = std::chrono::duration_cast<std::chrono::duration<double>>(t_after_calc - t_before_calc).count()*1000;
    time_check = std::chrono::duration_cast<std::chrono::duration<double>>(t_after_check - t_before_check).count()*1000;
#endif

	return bestpath;
}
