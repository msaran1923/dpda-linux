#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "DataAugmenter.h"

using namespace std;
using namespace cv;


class DataAugmenterFlip : public DataAugmenter {
public:
	DataAugmenterFlip(ImageLoader* imageLoader, ImageSaver* imageSaver)
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
			const bool applyFlip = (rand() % 100 < augmentationPercentage);

			if (applyFlip) {
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

		int flipCode = rand() % 3;
		flip(augmentedImage.clone(), augmentedImage, flipCode);
		
		if (pipelineDataAugmenter == 0)
			return augmentedImage;
		else
			return pipelineDataAugmenter->augmentImage(augmentedImage);
	}

private:

};
