#ifndef POINTS_VISUALIZATION_H
#define POINTS_VISUALIZATION_H
#include <vector>
#include "../cuda_quickhull/cuda_quickhull.cuh"

void drawPointsAndLines(const thrust::host_vector<cqh::Point>& points, const thrust::host_vector<cqh::Point>& hull);

#endif // POINTS_VISUALIZATION_H