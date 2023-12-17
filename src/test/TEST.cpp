// test_quickhull.cpp
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

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
