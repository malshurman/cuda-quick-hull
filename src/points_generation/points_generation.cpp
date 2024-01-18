#define _USE_MATH_DEFINES
#include "points_generation.h"
#include <random>
#include <cmath>

std::vector<cqh::Point> generateCirclePoints(std::mt19937& rng, std::uniform_real_distribution<double>& dist, int numPoints) {
    std::vector<cqh::Point> points;
    for (int i = 0; i < numPoints; i++) {
        double theta = 2 * M_PI * dist(rng);
        double r = sqrt(dist(rng));
        cqh::Point p;
        p.x = r * cos(theta);
        p.y = r * sin(theta);
        points.push_back(p);
    }
    return points;
}

std::vector<cqh::Point> generateTrianglePoints(std::mt19937& rng, std::uniform_real_distribution<double>& dist, int numPoints) {
    std::vector<cqh::Point> points;
    for (int i = 0; i < numPoints; i++) {
        double r1 = dist(rng);
        double r2 = dist(rng);
        cqh::Point p;
        p.x = (1 - sqrt(r1)) * 0 + (sqrt(r1) * (1 - r2)) * 1 + (sqrt(r1) * r2) * 0;
        p.y = (1 - sqrt(r1)) * 0 + (sqrt(r1) * (1 - r2)) * 0 + (sqrt(r1) * r2) * 1;
        points.push_back(p);
    }
    return points;
}

std::vector<cqh::Point> generateSquarePoints(std::mt19937& rng, std::uniform_real_distribution<double>& dist, int numPoints) {
    std::vector<cqh::Point> points;
    for (int i = 0; i < numPoints; i++) {
        cqh::Point p;
        p.x = dist(rng);
        p.y = dist(rng);
        points.push_back(p);
    }
    return points;
}

std::vector<cqh::Point> generatePoints(int seed, int numPoints, int numOutliers, float minCoordinate, float maxCoordinate, int outlierRangeOffset, Shape shape) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(minCoordinate, maxCoordinate);
    std::uniform_real_distribution<double> dist_outlier(minCoordinate - outlierRangeOffset, maxCoordinate + outlierRangeOffset);

    std::vector<cqh::Point> points;
    switch (shape) {
    case CIRCLE:
        points = generateCirclePoints(rng, dist, numPoints);
        break;
    case TRIANGLE:
        points = generateTrianglePoints(rng, dist, numPoints);
        break;
    case SQUARE:
        points = generateSquarePoints(rng, dist, numPoints);
        break;
    }

    for (int i = numPoints; i < numPoints + numOutliers; i++) {
        cqh::Point p;
        p.x = dist_outlier(rng);
        p.y = dist_outlier(rng);
        points.push_back(p);
    }

    return points;
}