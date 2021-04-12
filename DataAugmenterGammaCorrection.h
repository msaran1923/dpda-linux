#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "DataAugmenter.h"

using namespace std;
using namespace cv;


class DataAugmenterGammaCorrection : public DataAugmenter {
public:
	DataAugmenterGammaCorrection(ImageLoader* imageLoader, ImageSaver* imageSaver)
		: DataAugmenter(imageLoader, imageSaver)
	{

	}

	bool execute(String inputDirectory, String outputDirectory, path& imagePath,
		int augmentationCount, double scaleFactor, int augmentationPercentage)
	{
		String fileDirectory = imagePath.parent_path().string();
		String imageFileName = imagePath.stem().string();
		String extension = imagePath.extension().string();

		String inputFileName = fileDirectory + SLASH + imageFileName + extension;

		Mat image = imageLoader->load(inputFileName);
		if (image.empty()) {
			cout << endl;
			return false;
		}

		Mat resizedImage;
		resize(image, resizedImage, Size(), scaleFactor, scaleFactor);

		// write original image
		String outputFileName = fileDirectory + SLASH + imageFileName + extension;
		boost::replace_all(outputFileName, inputDirectory, outputDirectory);
		imageSaver->save(resizedImage, outputFileName);

		// write augmented images (x count)
		for (int k = 0; k < augmentationCount; k++) {
			const bool applyGammaCorrection = (rand() % 100 < augmentationPercentage);

			if (applyGammaCorrection) {
				stringstream ss;
				ss << fileDirectory << SLASH << imageFileName << "_" << k << extension;
				String outputFileName = ss.str();
				boost::replace_all(outputFileName, inputDirectory, outputDirectory);

				Mat augmentedImage = augmentImage(resizedImage);
				imageSaver->save(augmentedImage, outputFileName);
			}
		}

		return true;
	}

	Mat augmentImage(Mat image) {
		Mat augmentedImage = image.clone();

		double gamma = randRange(0.001, 2.5);

		double invGamma = 1.0 / gamma;

		Mat lookUpTable(1, 256, CV_8U);
		uchar* p = lookUpTable.ptr();
		for (int i = 0; i < 256; ++i) {
			p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
		}

		LUT(image, lookUpTable, augmentedImage);

		if (pipelineDataAugmenter == 0)
			return augmentedImage;
		else
			return pipelineDataAugmenter->augmentImage(augmentedImage);
	}

private:

};
