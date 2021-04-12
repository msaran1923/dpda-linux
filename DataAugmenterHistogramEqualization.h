#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "DataAugmenter.h"

using namespace std;
using namespace cv;


class DataAugmenterHistogramEqualization : public DataAugmenter {
public:
	DataAugmenterHistogramEqualization(ImageLoader* imageLoader, ImageSaver* imageSaver)
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
			const bool applyHistogramEqualization = (rand() % 100 < augmentationPercentage);

			if (applyHistogramEqualization) {
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

		if (image.channels() == 3) {
			Mat rgbChannels[3];
			split(image, rgbChannels);

			vector<Mat> equalizedImages;
			for (int i = 0; i < 3; i++) {
				Mat equalizedImage;
				equalizeHist(rgbChannels[i], equalizedImage);
				equalizedImages.push_back(equalizedImage);
			}
			merge(equalizedImages, augmentedImage);
		}
		if (image.channels() == 1) {
			equalizeHist(image, augmentedImage);
		}
		if (pipelineDataAugmenter == 0)
			return augmentedImage;
		else
			return pipelineDataAugmenter->augmentImage(augmentedImage);
	}
private:
};