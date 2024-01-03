#ifndef QUICKHULL_H
#define QUICKHULL_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace cqh {
	struct Point {
		double x;
		double y;
	};

	void computeConvexHull(const thrust::device_vector<cqh::Point>& input, thrust::device_vector<cqh::Point>& output);
}

#endif // QUICKHULL_H