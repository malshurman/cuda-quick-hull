#ifndef POINTS_GENERATOR_H
#define POINTS_GENERATOR_H

#include <vector>
#include "../cuda_quickhull/cuda_quickhull.cuh"

std::vector<cqh::Point> generatePoints(int num_points, int minRange, int maxRange);

#endif // POINTS_GENERATOR_H