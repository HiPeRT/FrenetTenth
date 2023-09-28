#include "../include/cubic_spline_planner.hpp"

// Python NONE equivalents :
// single variable -1e9
// Splines: http://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/16spline-curves.pdf

/************************
For a vector of points (x_i, y_i), it generates cubic splines 
between each (x_i, y_i) and (x_i+1, y_i+1) as :

f_i(x) = y_i(== a_i) + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3

where h_i = x_i - x_i-1
************************/


void printVecD(vecD A)
{
	cout << "\n------------Printing the vector----------\n";
	for(int i = 0; i < A.size(); i++)
		cout << A[i] << "\n";
	cout << "\n-----------------------------------------\n" << endl;
}


MatrixXd Spline::calc_A(vecD h) // get matrix A
{	
	MatrixXd A = MatrixXd::Zero(nx, nx);
	A(0, 0) = 1.0;
	for(int i = 0; i < nx - 1; i++)
	{
		if(i != nx - 2)
			A(i + 1, i + 1) = 2.0*(h[i] + h[i + 1]);
		A(i + 1, i) = h[i];
		A(i, i + 1) = h[i];
	}

	A(0, 1) = 0.0;
	A(nx - 1, nx - 2) = 0.0;
	A(nx - 1, nx - 1) = 1.0;

	return A;
}


MatrixXd Spline::calc_B(vecD h) // get matrix B
{
	MatrixXd B = MatrixXd::Zero(nx, 1);

	for(int i = 0; i < nx - 2; i++)
		B(i + 1, 0) = 3.0*(a[i + 2] - a[i + 1]) / h[i + 1] - 3.0*(a[i + 1] - a[i]) / h[i];

	return B;
}


int Spline::search_index(double p) // returns the index of the just greater element
{
	return upper_bound(x.begin(), x.end(), p) - x.begin() - 1;
}

//Modified with the approach used in:
//https://github.com/alexliniger/MPCC/blob/master/C%2B%2B/Spline/cubic_spline.cpp#L65
void Spline::init(vecD x_in, vecD y_in) // calculates the coeffs for splines
{

	Eigen::VectorXd xs, ys;


	xs = Map<VectorXd, Unaligned>(x_in.data(), x_in.size());
	ys = Map<VectorXd, Unaligned>(y_in.data(), y_in.size());
	nx = x_in.size();

	Eigen::VectorXd ax;
	Eigen::VectorXd bx;
	Eigen::VectorXd cx;
	Eigen::VectorXd dx;

	ax.setZero(nx);
	bx.setZero(nx-1);
	cx.setZero(nx);
	dx.setZero(nx-1);

	Eigen::VectorXd mu, h, alpha, l, z;
	mu.setZero(nx-1);
	h.setZero(nx-1);
	alpha.setZero(nx-1);
	l.setZero(nx);
	z.setZero(nx);


	ax = ys;

	for(int i = 0;i<nx-1;i++)
	{
		h(i) = xs(i+1) - xs(i);
	}


	for(int i = 1;i<nx-1;i++)
	{
		alpha(i) = 3.0/h(i)*(ax(i+1) - ax(i)) - 3.0/h(i-1)*(ax(i) - ax(i-1));
	}

	l(0) = 1.0;
	mu(0) = 0.0;
	z(0) = 0.0;
	for(int i = 1;i<nx-1;i++)
	{
		l(i) = 2.0*(xs(i+1) - xs(i-1)) - h(i-1)*mu(i-1);
		mu(i) = h(i)/l(i);
		z(i) = (alpha(i) - h(i-1)*z(i-1))/l(i);
	}
	l(nx-1) = 1.0;
	z(nx-1) = 0.0;

	cx(nx-1) = 0.0;

	for(int i = nx-2;i>=0;i--)
	{
		cx(i) = z(i) - mu(i)*cx(i+1);
		bx(i) = (ax(i+1) - ax(i))/h(i) - (h(i)*(cx(i+1) + 2.0*cx(i)))/3.0;
		dx(i) = (cx(i+1) - cx(i))/(3.0*h(i));
	}

	a = vector<double>(&ax[0], ax.data()+ax.cols()*ax.rows());
	b = vector<double>(&bx[0], bx.data()+bx.cols()*bx.rows());
	c = vector<double>(&cx[0], cx.data()+cx.cols()*cx.rows());
	d = vector<double>(&dx[0], dx.data()+dx.cols()*dx.rows());
	x = vector<double>(&xs[0], xs.data()+xs.cols()*xs.rows());
	y = vector<double>(&ys[0], ys.data()+ys.cols()*ys.rows());

}



double Spline::calc(double t) // find y at given x
{
	if(t < x[0])
		return NONE;
	else if(t > x[nx - 1])
		return NONE;
	int i = search_index(t);

	double dx = t - x[i]; 

	double result = a[i] + b[i]*dx + c[i]*dx*dx + d[i]*dx*dx*dx;

	return result;
}


double Spline::calcd(double t) // find y_dot at given x
{
	if(t < x[0])
		return NONE;
	else if(t > x[nx - 1])
		return NONE;
	
	int i = search_index(t);
	double dx = t - x[i];
	double result = b[i] + 2*c[i]*dx + 3*d[i]*dx*dx;

	return result;
}


double Spline::calcdd(double t) // y_doubleDot at given x
{
	if(t < x[0])
		return NONE;
	else if(t > x[nx - 1])
		return NONE;

	int i = search_index(t);

	double dx = t - x[i];
	double result = 2*c[i] + 6*d[i]*dx;

	return result;
}


/****************************
Spline 2D generates the parametric cubic spline 
of x and y as a function of s:

x = f(s)
y = g(s)
****************************/


vecD Spline2D::calc_s(vecD x, vecD y) // approximately calculates s along the spline 
{
	vecD dx;
	for(int i = 1; i < x.size(); i++)
		dx.push_back(x[i] - x[i - 1]);


	vecD dy;
	for(int i = 1; i < x.size(); i++)
		dy.push_back(y[i] - y[i - 1]);

	ds.clear(); 
	for(int i = 0; i < dx.size(); i++)
	{
		double temp;
		temp = sqrt(dx[i]*dx[i] + dy[i]*dy[i]);
		ds.push_back(temp);
	}

	vecD t;
	t.push_back(0);

	for(int i = 0; i < ds.size(); i++)
	{
		t.push_back(t.back() + ds[i]);
	}

	return t;
}


void Spline2D::calc_position(double *x, double *y, double t) // 
{
    double s = get_s_last();
    
    if (t <= 0)
        t = 0;
    while (t >= s)
        t -= s;
        
	*x = sx.calc(t);
	*y = sy.calc(t);
}

void Spline2D::calc_projection(double *s, double *d, double x, double y, double s_guess) // 
{

	double pos_x, pos_y;
	calc_position(&pos_x, &pos_y, s_guess);

	double s_opt = s_guess;
	double dist = sqrt(pow(pos_x - x, 2) + pow(pos_y - y, 2));

	if(dist >= 10)
	{
		std::cout << "----------------Distance: " << dist << std::endl;
	}

	double s_previous = s_opt;

	for(int i=0; i<30; ++i)
	{
		calc_position(&pos_x, &pos_y, s_opt);
		double dx = sx.calcd(s_opt);
		double ddx = sx.calcdd(s_opt);

		double dy = sy.calcd(s_opt);
		double ddy = sy.calcdd(s_opt);

		double diff_x = pos_x - x;
		double diff_y = pos_y - y;

		double jac = 2.0 * diff_x * dx + 2.0 * diff_y * dy;
        double hessian = 2.0 * dx * dx + 2.0 * diff_x * ddx +
                         2.0 * dy * dy + 2.0 * diff_y * ddy;
        // Newton method
        s_opt -= jac/hessian;
        s_opt = s_opt - this->get_s_last()*std::floor(s_opt/this->get_s_last());

        if(std::abs(s_previous - s_opt) <= 1e-5)
		{
			*s = s_opt;
		}

        s_previous = s_opt;
	}

	calc_position(&pos_x, &pos_y, *s);
	double diff_x = pos_x - x;
	double diff_y = pos_y - y;
	double yaw = calc_yaw(*s);
	*d = -diff_x*sin(yaw) + diff_y*cos(yaw);

}

double Spline2D::calc_curvature(double t)
{
    double s = get_s_last();
    
    if (t <= 0)
        t = 0;
    while (t >= s)
        t -= s;
        
	double dx = sx.calcd(t);
	double ddx = sx.calcdd(t);

	double dy = sy.calcd(t);
	double ddy = sy.calcdd(t);

	double k = (ddy*dx - ddx*dy) / (dx*dx  + dy *dy);

	return k;
}


double Spline2D::calc_yaw(double t)
{
    double s = get_s_last();
    
    if (t <= 0)
        t = 0;
    while (t >= s)
        t -= s;
        
	double dx = sx.calcd(t);
	double dy = sy.calcd(t);

	if(dx == 0)
		return 1.57*(dy > 0);
	double yaw = atan2(dy, dx);

	return yaw;
}


double Spline2D::get_s_last() 
{
	return s.back();
}


// generates the Spline2D with points along the spline at distance = ds, also returns yaw and curvature 
Spline2D calc_spline_course(vecD x, vecD y, vecD &rx, vecD &ry, vecD &ryaw, vecD &rk, double ds)
{
	Spline2D sp(x, y);

	vecD s;
	double sRange = sp.get_s_last();

	double sInc = 0;
	while(1)
	{
		if(sInc >= sRange)
			break;

		s.push_back(sInc);
		sInc = sInc + ds;
	}	
	//printVecD(s);
	for(int i = 0; i < s.size(); i++)
	{
		double ix, iy;
		sp.calc_position(&ix, &iy, s[i]);
		rx.push_back(ix);
		ry.push_back(iy);

		ryaw.push_back(sp.calc_yaw(s[i]));
		rk.push_back(sp.calc_curvature(s[i]));
	}

	return sp;
}
