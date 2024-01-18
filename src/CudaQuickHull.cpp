#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <map>
#include <iomanip>

#include "cuda_quickhull/cuda_quickhull.cuh"
#include "points_generation/points_generation.h"
#include "visualization/points_visualization.h"

int main() {
	/*
	// Open the CSV file
	std::string fullPathToFile("src/resources/horseshoe.csv");
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
		point.x = std::stod(x_str) * 500 + 400;
		point.y = std::stod(y_str) * 500 + 400;
		h_points.push_back(point);
	}

	std::cout << h_points.size() << " points read from CSV file!" << std::endl;

	int numPoints = h_points.size();
	*/
	const std::vector<int> numPointsList = { 50000000, 100000000 };
	std::map<int, std::pair<double, double>> results;

	for (const auto& numPoints : numPointsList) {
		const int numOutliers = 20;
		const int minRange = 200;
		const int maxRange = 800;
		const int outlierOffset = 100;
		// create a vector of points in the plane with thrust
		thrust::host_vector<cqh::Point> h_points = generatePoints(rand(), numPoints, numOutliers, minRange, maxRange, outlierOffset, SQUARE);
		std::cout << numPoints << " points randomly generated!" << std::endl;

		std::cout << "Starting..." << std::endl;

		double totalDurationCV = 0;
		double totalDurationCQH = 0;
		size_t numRuns = 15;

		for (size_t i = 0; i < numRuns; i++) {
			std::vector<cv::Point> cvPoints;
			std::vector<cv::Point> cvHull;

			thrust::device_vector<cqh::Point> d_points = h_points;
			thrust::device_vector<cqh::Point> hull;

			for (const auto& point : h_points) {
				cvPoints.push_back(cv::Point(point.x, point.y));
			}

			auto startCV = std::chrono::high_resolution_clock::now();
			cv::convexHull(cvPoints, cvHull);
			auto endCV = std::chrono::high_resolution_clock::now();

			auto durationCV = std::chrono::duration_cast<std::chrono::milliseconds>(endCV - startCV).count();
			totalDurationCV += durationCV;
			std::cout << "cv::convexHull execution time: " << durationCV << " ms" << std::endl;

			auto startCQH = std::chrono::high_resolution_clock::now();
			cqh::computeConvexHull(d_points, hull);
			auto endCQH = std::chrono::high_resolution_clock::now();

			auto durationCQH = std::chrono::duration_cast<std::chrono::milliseconds>(endCQH - startCQH).count();
			totalDurationCQH += durationCQH;
			std::cout << "cqh::computeConvexHull execution time: " << durationCQH << " ms" << std::endl;
		}

		double averageDurationCV = totalDurationCV / numRuns;
		double averageDurationCQH = totalDurationCQH / numRuns;
		std::cout << std::endl << std::endl << "Results for " << numPoints << " points randomly generated:" << std::endl;
		std::cout << "Average cv::convexHull execution time over " << numRuns << " runs: " << averageDurationCV << " ms" << std::endl;
		std::cout << "Average cqh::computeConvexHull execution time over " << numRuns << " runs: " << averageDurationCQH << " ms" << std::endl;
		results[numPoints] = std::make_pair(averageDurationCV, averageDurationCQH);
	}

	// Print the results in a table
	std::cout << "| Number of Points | Average cv::convexHull Time (ms) | Average cqh::computeConvexHull Time (ms) |" << std::endl;
	std::cout << "|------------------|----------------------------------|------------------------------------------|" << std::endl;
	for (const auto& result : results) {
		std::cout << "| " << std::setw(16) << result.first << " | " << std::setw(32) << result.second.first << " | " << std::setw(40) << result.second.second << " |" << std::endl;
	}

	/*
	// Verify
	std::vector<cv::Point> cvPoints;
	std::vector<cv::Point> cvHull;
	for (const auto& point : h_points) {
		cvPoints.push_back(cv::Point(point.x, point.y));
	}
	cv::convexHull(cvPoints, cvHull);

	thrust::device_vector<cqh::Point> d_points = h_points;
	thrust::device_vector<cqh::Point> hull;
	cqh::computeConvexHull(d_points, hull);
	thrust::host_vector<cqh::Point> h_out = hull;
	*/

	/*
	for (size_t i = 0; i < cvHull.size(); i++) {
		std::cout << "(" << cvHull[i].x << "," << cvHull[i].y << "), (" << h_out[i].x << "," << h_out[i].y << ")" << std::endl;
	}

	// Check if both methods return the same output
	// Compare cvHull and h_out
	bool isEqual = true;
	if (cvHull.size() != h_out.size()) {
		isEqual = false;
	}
	else {
		for (size_t i = 0; i < cvHull.size(); i++) {
			std::cout << "(" << cvHull[i].x << "," << cvHull[i].y << "), (" << h_out[i].x << "," << h_out[i].y << ")" << std::endl;
			if (cvHull[i].x != h_out[i].x || cvHull[i].y != h_out[i].y) {
				isEqual = false;
				break;
			}
		}
	}
	std::cout << "Both methods return the same output: " << (isEqual ? "Yes" : "No") << std::endl;
	*/
	//drawPointsAndLines(h_points, h_out);


	return 0;
}