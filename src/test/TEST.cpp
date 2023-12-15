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

    // Initialize your points (you can use random or known data)
    // ...

    // Execute the quick hull algorithm
    quickHullCUDA(points, N, hullPoints, &numHullPoints);

    // Add your assertions here to check correctness

    // Cleanup
    free(points);
    free(hullPoints);
}
