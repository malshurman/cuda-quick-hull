// quickhull.h
#pragma once

struct Point;

void quickHullCUDA(Point* points, int numPoints, int* hullPoints, int* numHullPoints);