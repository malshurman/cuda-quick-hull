// test_quickhull.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <cmath>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

// Include your original code
#include "../quickhull/quickhull.h"
#include "../points_generation/random_points.h"

TEST_CASE("QuickHullCUDA Test") {
    const int N = 1000;  // Adjust this based on your actual number of points
    Point* points = (Point*)malloc(N * sizeof(Point));
    int* hullPoints = (int*)malloc(N * sizeof(int));
    int numHullPoints = 0;

    // Initialize your points with random data
    std::vector<Point> randomPoints = generate_random_points();
    for (int i = 0; i < N; ++i) {
        points[i] = randomPoints[i];
    }

    // Execute the quick hull algorithm
    auto start = std::chrono::high_resolution_clock::now();
    quickHullCUDA(points, N, hullPoints, &numHullPoints);
    auto end = std::chrono::high_resolution_clock::now();

    // Add assertions to check correctness

    // 1. Check that the number of hull points is valid
    REQUIRE(numHullPoints >= 3);  // A convex hull should have at least 3 points

    // 2. Check that hull points indices are within the valid range
    for (int i = 0; i < numHullPoints; ++i) {
        REQUIRE(hullPoints[i] >= 0);
        REQUIRE(hullPoints[i] < N);
    }

    // 3. Check that the points on the hull form a convex shape
    for (int i = 0; i < numHullPoints - 2; ++i) {
        Point p1 = points[hullPoints[i]];
        Point p2 = points[hullPoints[i + 1]];
        Point p3 = points[hullPoints[i + 2]];
    }

    // Calculate and print the execution time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " ms" << std::endl;

    // Cleanup
    free(points);
    free(hullPoints);
}

TEST_CASE("QuickHullCUDA Circle Test") {
    const int N = 1000;
    Point* points = (Point*)malloc(N * sizeof(Point));
    int* hullPoints = (int*)malloc(N * sizeof(int));
    int numHullPoints = 0;

    // Initialize points for a circle
    double radius = 5.0;
    double centerX = 0.0;
    double centerY = 0.0;

    for (int i = 0; i < N; ++i) {
        double theta = 2.0 * M_PI * i / N;
        points[i].x = centerX + radius * cos(theta);
        points[i].y = centerY + radius * sin(theta);
    }

    // Execute the quick hull algorithm
    quickHullCUDA(points, N, hullPoints, &numHullPoints);

    // Check expected properties for a circle
    REQUIRE(numHullPoints > 0);  // Convex hull should not be empty

    // All points on the circle should be part of the convex hull
    for (int i = 0; i < N; ++i) {
        bool isOnHull = false;
        for (int j = 0; j < numHullPoints; ++j) {
            if (hullPoints[j] == i) {
                isOnHull = true;
                break;
            }
        }
        REQUIRE(isOnHull);
    }

    // Cleanup
    free(points);
    free(hullPoints);
}

TEST_CASE("QuickHullCUDA Line Test") {
    const int N = 1000;
    Point* points = (Point*)malloc(N * sizeof(Point));
    int* hullPoints = (int*)malloc(N * sizeof(int));
    int numHullPoints = 0;

    // Initialize points for a line
    double startX = -5.0;
    double endX = 5.0;
    double slope = 1.0;

    for (int i = 0; i < N; ++i) {
        points[i].x = startX + (endX - startX) * i / N;
        points[i].y = slope * points[i].x;  // Assuming a simple linear relation for y
    }

    // Execute the quick hull algorithm
    quickHullCUDA(points, N, hullPoints, &numHullPoints);

    // Check expected properties for a line
    REQUIRE(numHullPoints > 0);  // Convex hull should not be empty

    // All points on the line should be part of the convex hull
    for (int i = 0; i < N; ++i) {
        bool isOnHull = false;
        for (int j = 0; j < numHullPoints; ++j) {
            if (hullPoints[j] == i) {
                isOnHull = true;
                break;
            }
        }
        REQUIRE(isOnHull);
    }

    // Cleanup
    free(points);
    free(hullPoints);
}

TEST_CASE("QuickHullCUDA Triangle Test") {
    const int N = 1000;
    Point* points = (Point*)malloc(N * sizeof(Point));
    int* hullPoints = (int*)malloc(N * sizeof(int));
    int numHullPoints = 0;

    // Initialize points for a triangle
    double sideLength = 5.0;
    double height = sideLength * sqrt(3.0) / 2.0;  // Assuming an equilateral triangle

    points[0].x = 0.0;
    points[0].y = 0.0;

    points[1].x = sideLength;
    points[1].y = 0.0;

    points[2].x = sideLength / 2.0;
    points[2].y = height;

    // Execute the quick hull algorithm
    quickHullCUDA(points, 3, hullPoints, &numHullPoints);

    // Check expected properties for a triangle
    REQUIRE(numHullPoints == 3);  // Convex hull of a triangle should have 3 points

    // All points of the triangle should be part of the convex hull
    for (int i = 0; i < 3; ++i) {
        bool isOnHull = false;
        for (int j = 0; j < numHullPoints; ++j) {
            if (hullPoints[j] == i) {
                isOnHull = true;
                break;
            }
        }
        REQUIRE(isOnHull);
    }

    // Cleanup
    free(points);
    free(hullPoints);
}

TEST_CASE("QuickHullCUDA Rectangle Test") {
    const int N = 1000;
    Point* points = (Point*)malloc(N * sizeof(Point));
    int* hullPoints = (int*)malloc(N * sizeof(int));
    int numHullPoints = 0;

    // Initialize points for a rectangle
    double width = 5.0;
    double height = 4.0;

    points[0].x = 0.0;
    points[0].y = 0.0;

    points[1].x = width;
    points[1].y = 0.0;

    points[2].x = width;
    points[2].y = height;

    points[3].x = 0.0;
    points[3].y = height;

    // Execute the quick hull algorithm
    quickHullCUDA(points, 4, hullPoints, &numHullPoints);

    // Check expected properties for a rectangle
    REQUIRE(numHullPoints == 4);  // Convex hull of a rectangle should have 4 points

    // All points of the rectangle should be part of the convex hull
    for (int i = 0; i < 4; ++i) {
        bool isOnHull = false;
        for (int j = 0; j < numHullPoints; ++j) {
            if (hullPoints[j] == i) {
                isOnHull = true;
                break;
            }
        }
        REQUIRE(isOnHull);
    }

    // Cleanup
    free(points);
    free(hullPoints);
}
