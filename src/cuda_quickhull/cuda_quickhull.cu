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

__host__
void quickHull(const thrust::device_vector<cqh::Point>& v, const cqh::Point& a, const cqh::Point& b, thrust::device_vector<cqh::Point>& hull) {
    if (v.empty()) {
        return;
    }

    cqh::Point f = *thrust::max_element(v.begin(), v.end(), DistanceFromLine(a, b));

    thrust::device_vector<cqh::Point> aboveAFsegment(v.size());
    size_t aboveAFSize = thrust::copy_if(thrust::device, v.begin(), v.end(), aboveAFsegment.begin(), isAboveLine(a, f)) - aboveAFsegment.begin();
    aboveAFsegment.resize(aboveAFSize);
    thrust::device_vector<cqh::Point> aboveFBsegment(v.size());
    size_t aboveFBSize = thrust::copy_if(thrust::device, v.begin(), v.end(), aboveFBsegment.begin(), isAboveLine(f, b)) - aboveFBsegment.begin();
    aboveFBsegment.resize(aboveFBSize);
    hull.push_back(f);
    quickHull(aboveAFsegment, a, f, hull);
    quickHull(aboveFBsegment, f, b, hull);
}

__host__
void quickHull(const thrust::device_vector<cqh::Point>& input, thrust::device_vector<cqh::Point>& output) {
    cqh::Point pointWithMinX = *thrust::min_element(input.begin(), input.end(), PointComparatorByX());
    cqh::Point pointWithMaxX = *thrust::max_element(input.begin(), input.end(), PointComparatorByX());
    thrust::device_vector<cqh::Point> aboveLine(input.size());
    size_t aboveSize = thrust::copy_if(thrust::device, input.begin(), input.end(), aboveLine.begin(), isAboveLine(pointWithMinX, pointWithMaxX)) - aboveLine.begin();
    aboveLine.resize(aboveSize);
    thrust::device_vector<cqh::Point> belowLine(input.size());
    size_t belowSize = thrust::copy_if(thrust::device, input.begin(), input.end(), belowLine.begin(), isAboveLine(pointWithMaxX, pointWithMinX)) - belowLine.begin();
    belowLine.resize(belowSize);
    output.push_back(pointWithMinX);
    quickHull(aboveLine, pointWithMinX, pointWithMaxX, output);
    output.push_back(pointWithMaxX);
    quickHull(belowLine, pointWithMaxX, pointWithMinX, output);
}

void cqh::computeConvexHull(const thrust::device_vector<cqh::Point>& input, thrust::device_vector<cqh::Point>& output) {
    quickHull(input, output);
}