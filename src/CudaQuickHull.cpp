#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "cuda_quickhull/cuda_quickhull.cuh"
#include "points_generation/points_generation.h"
#include "visualization/points_visualization.h"

int main() {
	/*
	// Open the CSV file
	std::string fullPathToFile("src/resources/generisane_tacke_zvijezda.csv");
	std::cout << "Attempting to open file at: " << fullPathToFile << std::endl;
	std::ifstream file(fullPathToFile);

	// Check if the file is open
	if (!file.is_open()) {
		std::cout << "Could not open file" << std::endl;
		return 1;
	}

	// Skip the first line (header)
	std::string line;
	std::getline(file, line);

	// Create a vector of points
	thrust::host_vector<cqh::Point> h_points;

	// Read the points from the CSV file
	while (std::getline(file, line)) {
		std::istringstream ss(line);
		std::string x_str, y_str;

		// Get the x and y coordinates
		std::getline(ss, x_str, ',');
		std::getline(ss, y_str, ',');

		// Convert the coordinates to double and add the point to the vector
		cqh::Point point;
		point.x = std::stod(x_str) * 500 + 500;
		point.y = std::stod(y_str) * 500 + 500;
		h_points.push_back(point);
	}

	std::cout << h_points.size() << " points read from CSV file!" << std::endl;
	*/


	const int N = 20000;
	// create a vector of points in the plane with thrust
	thrust::host_vector<cqh::Point> h_points = generatePoints(N, 100, 900);
	std::cout << N << " points randomly generated!" << std::endl;

	// create device vector
	thrust::device_vector<cqh::Point> d_points = h_points;
	thrust::device_vector<cqh::Point> hull;

	std::cout << "Starting..." << std::endl;

	double totalDuration = 0;
	size_t numRuns = 3;

	for (size_t i = 0; i < numRuns; i++) {
		std::vector<cv::Point> cvPoints;
		std::vector<cv::Point> cvHull;

		for (const auto& point : h_points) {
			cvPoints.push_back(cv::Point(point.x, point.y));
		}

		auto start = std::chrono::high_resolution_clock::now();
		//cqh::computeConvexHull(d_points, hull);
		cv::convexHull(cvPoints, cvHull);
		auto end = std::chrono::high_resolution_clock::now();

		// Compute time interval
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		totalDuration += duration;
		std::cout << "Execution time: " << duration << " ms" << std::endl;
	}

	double averageDuration = totalDuration / numRuns;
	std::cout << "Average execution time over " << numRuns << " runs: " << averageDuration << " ms" << std::endl;

	auto start = std::chrono::high_resolution_clock::now();
	cqh::computeConvexHull(d_points, hull);
	auto end = std::chrono::high_resolution_clock::now();
	thrust::host_vector<cqh::Point> h_out = hull;


	drawPointsAndLines(h_points, h_out);


	return 0;
}