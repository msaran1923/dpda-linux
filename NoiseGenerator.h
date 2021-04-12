#pragma once
#include "OSDefines.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


class NoiseGenerator {
public:
	virtual Mat create(int width, int height, double roughness) = 0;

private:

};
