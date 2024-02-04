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

int drawConvexHullForCSV(thrust::host_vector<cqh::Point>& h_points, const std::string& fileName) {
	std::cout << "Attempting to open file at: " << fileName << std::endl;

	// Append directory path and file extension to the filename
	std::string fullPathToFile = "src/resources/" + fileName + ".csv";

	std::ifstream file(fullPathToFile);

	// Check if the file is open
	if (!file.is_open()) {
		std::cout << "Could not open file" << std::endl;
		return 1;
	}

	// Skip the first line (header)
	std::string line;
	std::getline(file, line);

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
	thrust::device_vector<cqh::Point> d_points = h_points;
	thrust::device_vector<cqh::Point> hull;

	std::cout << "Starting..." << std::endl;

	auto startCQH = std::chrono::high_resolution_clock::now();
	cqh::computeConvexHull(d_points, hull);
	auto endCQH = std::chrono::high_resolution_clock::now();

	auto durationCQH = std::chrono::duration_cast<std::chrono::milliseconds>(endCQH - startCQH).count();
	std::cout << "cqh::computeConvexHull execution time: " << durationCQH << " ms" << std::endl;

	drawPointsAndLines(h_points, hull);
}

void drawConvexHullForRandomPoints(int numPoints) {
	const int numPointsReduced = numPoints * 0.99;
	const int numOutliers = numPoints * 0.01;
	const int minRange = 200;
	const int maxRange = 800;
	const int outlierOffset = 100;
	thrust::host_vector<cqh::Point> h_points = generatePoints(rand(), numPointsReduced, numOutliers, minRange, maxRange, outlierOffset, SQUARE);
	std::cout << numPointsReduced << " points and " << numOutliers << " outliers randomly generated!" << std::endl;

	thrust::device_vector<cqh::Point> d_points = h_points;
	thrust::device_vector<cqh::Point> hull;

	auto startCQH = std::chrono::high_resolution_clock::now();
	cqh::computeConvexHull(d_points, hull);
	auto endCQH = std::chrono::high_resolution_clock::now();

	auto durationCQH = std::chrono::duration_cast<std::chrono::milliseconds>(endCQH - startCQH).count();
	std::cout << "cqh::computeConvexHull execution time: " << durationCQH << " ms" << std::endl;

	drawPointsAndLines(h_points, hull);
}

int compareSequentialAndParallelConvexHull() {
	const std::vector<long long> numPointsList = { 10000, 50000, 100000, 1000000, 5000000, 10000000 };
	std::map<int, std::pair<double, double>> results;

	for (const auto& numPoints : numPointsList) {
		const int numPointsReduced = numPoints * 0.9;
		const int numOutliers = numPoints * 0.1;
		const int minRange = 200;
		const int maxRange = 800;
		const int outlierOffset = 100;
		// create a vector of points in the plane with thrust
		thrust::host_vector<cqh::Point> h_points = generatePoints(rand(), numPointsReduced, numOutliers, minRange, maxRange, outlierOffset, SQUARE);
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

	return 0;
}

int main() {
	int choice;
	std::cout << "Please choose an option:\n";
	std::cout << "1. Calculate convex hull for points read from a CSV file\n";
	std::cout << "2. Calculate convex hull for random points\n";
	std::cout << "3. Run a predetermined test to compare GPU and CPU convex hull\n";
	std::cin >> choice;

	thrust::host_vector<cqh::Point> h_points;

	switch (choice) {
	case 1: {
		std::string filename;
		std::cout << "Enter the name of the CSV file: ";
		std::cin >> filename;
		int numPoints = drawConvexHullForCSV(h_points, filename);
		break;
	}
	case 2: {
		int numPoints;
		std::cout << "Enter the number of random points: ";
		std::cin >> numPoints;
		drawConvexHullForRandomPoints(numPoints);
		break;
	}
	case 3: {
		compareSequentialAndParallelConvexHull();
		break;
	}
	default:
		std::cout << "Invalid option. Please enter 1, 2, or 3.\n";
		break;
	}

	return 0;
}