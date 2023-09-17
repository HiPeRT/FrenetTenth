#ifndef STRUCTURES_HPP
#define STRUCTURES_HPP

#include <vector>

struct obstacle{
    double x, y, radius;
    double s, d, t;
};

// This can be used for a generic rectangular bounding box
typedef struct ObjectStruct {
    unsigned int    object_id;
    double          x;
    double          y;
    double          z;
    double          heading;
    double          velocity;
    double   		length;
    double   		width;

    double s, d, t;
    double local_heading;
} ObjectStruct;

template <class T>
struct FrenetPath{
    std::vector<T> t, d, d_d, d_dd, d_ddd, s, s_d, s_dd, s_ddd;
    std::vector<T> x, y, yaw, ds, c;
    T cd, cv, cf; //costs
    bool empty = false; //true if no path available
};

#endif