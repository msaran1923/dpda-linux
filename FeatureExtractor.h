#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "RGB.h"

using namespace cv;


class FeatureExtractor {
public:
	static Mat create(Mat& image) 
	{
		// create pyramid
		vector<Mat> pyramidImages;

		int pyramidLevelCount = 4; // 1;
		int n = 0;
		for (int k = 0; k < pyramidLevelCount; k++) {
			double scaleFactor = pow(2, -k);

			Mat pyramidImage;
			if (k == 0)
				pyramidImage = image;
			else
				resize(image, pyramidImage, Size(), scaleFactor, scaleFactor, INTER_LANCZOS4);

			pyramidImages.push_back(pyramidImage);

			n += pyramidImage.cols * pyramidImage.rows;
		}


		// create features using images in the pyramid
		Mat features(n, image.channels(), CV_32F);

		int index = 0;
		for (int i = 0; i < pyramidImages.size(); i++) {
			Mat& pyramidImage = pyramidImages.at(i);

			for (int y = 0; y < pyramidImage.rows; y++) {
				RGB<uchar>* irow = (RGB<uchar>*)(pyramidImage.data + y * pyramidImage.step);

				for (int x = 0; x < pyramidImage.cols; x++) {
					RGB<float>* fdata = (RGB<float>*)(features.data + index * features.step);

					*fdata = irow[x];		// assigns red, blue and green via operator=

					index++;
				}
			}
		}

		return features;
	}

};
