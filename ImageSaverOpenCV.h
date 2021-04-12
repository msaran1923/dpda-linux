#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "ImageSaver.h"

using namespace cv;


class ImageSaverOpenCV : public ImageSaver {
public:
	void save(const Mat& image, string outputFileName)
	{
		imwrite(outputFileName, image);
	}

};
