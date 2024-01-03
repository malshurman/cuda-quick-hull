#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "cuda_quickhull/cuda_quickhull.cuh"
#include "points_generation/points_generation.h"
#include "visualization/points_visualization.h"

int main() {
	const int N = 500;
	// create a vector of points in the plane with thrust
	thrust::host_vector<cqh::Point> h_points = generatePoints(N, 100, 900);
	std::cout << N << " points randomly generated!" << std::endl;

	// create device vector
	thrust::device_vector<cqh::Point> d_points = h_points;
	thrust::device_vector<cqh::Point> hull;

	std::cout << "Starting..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	cqh::computeConvexHull(d_points, hull);
	auto end = std::chrono::high_resolution_clock::now();
	thrust::host_vector<cqh::Point> h_out = hull;

	std::cout << "Found convex hull with " << h_out.size() << " points.\nPrinting the first and last three points of the hull:" << std::endl;

	// Print first 3 elements
	for (int i = 0; i < 3; i++) {
		printf("(%f, %f)\n", h_out[i].x, h_out[i].y);
	}

	std::cout << ".\n.\n." << std::endl;

	// Print last 3 elements
	for (size_t i = h_out.size() - 3; i < h_out.size(); i++) {
		printf("(%f, %f)\n", h_out[i].x, h_out[i].y);
	}

	// Compute time interval
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "Execution time: " << duration << " ms" << std::endl;

	std::vector<cqh::Point> cvHull;
	std::vector<cqh::Point> cvPoints;
	for (size_t i = 0; i < h_points.size(); i++) {
		cqh::Point point;
		point.x = h_points[i].x;
		point.y = h_points[i].y;
		cvPoints.push_back(point);
	}
	//cv::convexHull(cvPoints, cvHull);

	drawPointsAndLines(h_points, h_out);


	return 0;
}