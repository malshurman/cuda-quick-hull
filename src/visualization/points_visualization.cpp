#include <opencv2/opencv.hpp>
#include "points_visualization.h"

void drawPointsAndLines(const thrust::host_vector<cqh::Point>& points, const thrust::host_vector<cqh::Point>& hull) {
    cv::Mat img(1000, 1000, CV_8UC3, cv::Scalar(200, 200, 200));

    for (const auto& point: points) {
        cv::circle(img, cv::Point(point.x, point.y), 3, cv::Scalar(0, 0, 0), -1);
    }

    for (size_t i = 0; i < hull.size() - 1; ++i) {
        cv::line(img, cv::Point(hull[i].x, hull[i].y), cv::Point(hull[i + 1].x, hull[i + 1].y), cv::Scalar(0, 0, 255), 2);
    }

    cv::line(img, cv::Point(hull.back().x, hull.back().y), cv::Point(hull.front().x, hull.front().y), cv::Scalar(0, 0, 255), 2);

    cv::imshow("Convex Hull", img);
    cv::waitKey(0);
}