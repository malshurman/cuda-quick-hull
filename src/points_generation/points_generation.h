#ifndef POINTS_GENERATOR_H
#define POINTS_GENERATOR_H

#include <vector>
#include "../cuda_quickhull/cuda_quickhull.cuh"

enum Shape {
	CIRCLE,
	TRIANGLE,
	SQUARE
};

std::vector<cqh::Point> generatePoints(int seed, int numPoints, int numOutliers, float minRange, float maxRange, int outlierRangeOffset, Shape shape);

#endif // POINTS_GENERATOR_H