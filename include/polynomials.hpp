#ifndef POLYNOMIALS_HPP
#define POLYNOMIALS_HPP

#include <math.h> 
#include <eigen3/Eigen/Dense>
#include <cmath>

using namespace Eigen;

template <class T>
class quintic{
private:
	T xs, vxs, axs, xe, vxe, axe, a0, a1, a2, a3, a4, a5;
public:
	quintic(T, T, T, T, T, T, T);
	T calc_point(T);
	T calc_first_derivative(T);
	T calc_second_derivative(T);
	T calc_third_derivative(T);
};

template <class T>
class quartic{
private:
	T xs, vxs, axs, vxe, axe, a0, a1, a2, a3, a4;
public:
	quartic(T, T, T, T, T, T);
	T calc_point(T);
	T calc_first_derivative(T);
	T calc_second_derivative(T);
	T calc_third_derivative(T);
};

template class quintic<double>;
template class quintic<float>;
#if defined(__x86_64__) || defined(_M_X64)
typedef float __fp16;
#else
template class quintic<__fp16>;
#endif

template class quartic<double>;
template class quartic<float>;
#if defined(__x86_64__) || defined(_M_X64)
typedef float __fp16;
#else
template class quartic<__fp16>;
#endif

#endif
