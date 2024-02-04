#include "cuda_quickhull.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

struct PointComparatorByX {
    __device__
        bool operator()(const cqh::Point& p1, const cqh::Point& p2) {
        return p1.x < p2.x;
    }
};

struct isAboveLine {
private:
    const cqh::Point p;
    const cqh::Point q;
public:
    __host__ __device__
        isAboveLine(const cqh::Point& p, const cqh::Point& q) :p{ p }, q{ q } {};

    __device__
        bool operator()(const cqh::Point& point) {
        return (((q.x - p.x) * (point.y - p.y)) - ((point.x - p.x) * (q.y - p.y))) > 1e-8;
    }
};

struct DistanceFromLine {
private:
    double a;
    double b;
    double c;

public:
    __host__ __device__
        DistanceFromLine(const cqh::Point& p, const cqh::Point& q) : a(q.y - p.y), b(p.x - q.x), c((q.x* p.y) - (p.x * q.y)) {}

    __device__
        bool operator()(const cqh::Point& p1, const cqh::Point& p2) {
        auto d1 = fabs((a * p1.x + b * p1.y + c) / std::sqrt(a * a + b * b));
        auto d2 = fabs((a * p2.x + b * p2.y + c) / std::sqrt(a * a + b * b));
        return d1 < d2;
    }
};

struct GeometryUtils {
    __host__ __device__
    static int orientation(const cqh::Point& p, const cqh::Point& q, const cqh::Point& r) {
        int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);


        if (val == 0) return 0;  // colinear
        return (val > 0) ? 1 : 2; // clock or counterclock wise
    }


    __host__ __device__
    static float distanceSq(const cqh::Point& p1, const cqh::Point& p2) {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
    }
};


struct PolarOrderComparator {
    cqh::Point leftmostPoint;


    __host__ __device__
    PolarOrderComparator(const cqh::Point& leftmostPoint) : leftmostPoint(leftmostPoint) {}


    __host__ __device__
    bool operator()(const cqh::Point& p1, const cqh::Point& p2) const {
        int order = GeometryUtils::orientation(leftmostPoint, p1, p2);
        if (order == 0) {
            return GeometryUtils::distanceSq(leftmostPoint, p1) < GeometryUtils::distanceSq(leftmostPoint, p2);
        }
        return (order == 2);
    }
};


__host__
void quickHullIterative(const thrust::device_vector<cqh::Point>& v, const cqh::Point& a, const cqh::Point& b, thrust::device_vector<cqh::Point>& hull) {
    // The algorithm finishes when there are no more points left to consider
    if (v.empty()) {
        return;
    }

    // We find the point furthest away from the line a-b. This forms a triangle, we only consider points that are outside the triangle.
    cqh::Point f = *thrust::max_element(v.begin(), v.end(), DistanceFromLine(a, b));

    // We only look at points outside of the triangle, which corresponds to being above the lines a-f and f-b
    thrust::device_vector<cqh::Point> aboveAFsegment(v.size());
    size_t aboveAFSize = thrust::copy_if(thrust::device, v.begin(), v.end(), aboveAFsegment.begin(), isAboveLine(a, f)) - aboveAFsegment.begin();
    aboveAFsegment.resize(aboveAFSize);
    thrust::device_vector<cqh::Point> aboveFBsegment(v.size());
    size_t aboveFBSize = thrust::copy_if(thrust::device, v.begin(), v.end(), aboveFBsegment.begin(), isAboveLine(f, b)) - aboveFBsegment.begin();
    aboveFBsegment.resize(aboveFBSize);

    // Again, we add the point that is in the convex hull and continue searching for points recursively
    hull.push_back(f);

    quickHullIterative(aboveAFsegment, a, f, hull);
    quickHullIterative(aboveFBsegment, f, b, hull);
}

__host__
void quickHullStart(const thrust::device_vector<cqh::Point>& input, thrust::device_vector<cqh::Point>& output) {
    // Get leftmost and rightmost point
    cqh::Point pointWithMinX = *thrust::min_element(input.begin(), input.end(), PointComparatorByX());
    cqh::Point pointWithMaxX = *thrust::max_element(input.begin(), input.end(), PointComparatorByX());

    // Split all points into those above and below the line made by connecting leftmost and rightmost point
    thrust::device_vector<cqh::Point> aboveLine(input.size());
    size_t aboveSize = thrust::copy_if(thrust::device, input.begin(), input.end(), aboveLine.begin(), isAboveLine(pointWithMinX, pointWithMaxX)) - aboveLine.begin();
    aboveLine.resize(aboveSize);
    thrust::device_vector<cqh::Point> belowLine(input.size());
    size_t belowSize = thrust::copy_if(thrust::device, input.begin(), input.end(), belowLine.begin(), isAboveLine(pointWithMaxX, pointWithMinX)) - belowLine.begin();
    belowLine.resize(belowSize);

    // The leftmost and rightmost points are definitely in the convex hull, now we recursively find the rest
    output.push_back(pointWithMinX);
    quickHullIterative(aboveLine, pointWithMinX, pointWithMaxX, output);

    output.push_back(pointWithMaxX);
    quickHullIterative(belowLine, pointWithMaxX, pointWithMinX, output);
}

void cqh::computeConvexHull(const thrust::device_vector<cqh::Point>& input, thrust::device_vector<cqh::Point>& output) {
    quickHullStart(input, output);

    // Find the leftmost point
    Point leftmostPoint = *thrust::min_element(output.begin(), output.end(), PointComparatorByX());

    // Sort the points in counterclockwise order
    thrust::sort(thrust::device, output.begin(), output.end(), PolarOrderComparator(leftmostPoint));
}