#include "points_generation.h"
#include <random>

static unsigned int SEED = 2025;

std::vector<cqh::Point> generatePoints(int num_points, int minRange, int maxRange) {
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<double> dist(minRange, maxRange);

    std::vector<cqh::Point> points(num_points);
    for (int i = 0; i < num_points; i++) {
        points[i].x = dist(rng);
        points[i].y = dist(rng);
    }

    return points;
}