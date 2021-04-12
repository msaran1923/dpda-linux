#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "ImageLoader.h"

using namespace cv;


class ImageLoaderOpenCV : public ImageLoader {
public:
	Mat load(path imagePath)
	{
		String inputFileName = getFileName(imagePath);

		Mat image = imread(inputFileName, cv::IMREAD_COLOR);

		if (image.empty()) {
			addImageLoadError(imagePath);
		}

		return image;
	}

};
