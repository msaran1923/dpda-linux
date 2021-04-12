#pragma once
#include "OSDefines.h"
#include "DataAugmenter.h"

using namespace std;


class DataAugmenterRandomErase : public DataAugmenter {
public:
	DataAugmenterRandomErase(ImageLoader* imageLoader, ImageSaver* imageSaver)
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
			const bool applyRandomErase = (rand() % 100 < augmentationPercentage);

			if (applyRandomErase) {
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

		int minPatchWidth = (int)(0.05 * image.cols);
		int minPatchHeight = (int)(0.05 * image.rows);
		int maxPatchWidth = (int)(0.50 * image.cols);
		int maxPatchHeight = (int)(0.50 * image.rows);

		const int randomWidth = max(rand() % maxPatchWidth, minPatchWidth);
		const int randomHeight = max(rand() % maxPatchHeight, minPatchHeight);

		int x1 = max(rand() % (image.cols - minPatchWidth), 1);
		int y1 = max(rand() % (image.rows - minPatchHeight), 1);
		int x2 = min(x1 + randomWidth, image.cols - 1);
		int y2 = min(y1 + randomHeight, image.rows - 1);

		for (int y = y1; y < y2; y++) {
			RGB<uchar>* irow = (RGB<uchar>*)(augmentedImage.data + y * augmentedImage.step);

			for (int x = x1; x < x2; x++) {
				irow[x].red = 0;
				irow[x].green = 0;
				irow[x].blue = 0;
			}
		}

		if (pipelineDataAugmenter == 0)
			return augmentedImage;
		else
			return pipelineDataAugmenter->augmentImage(augmentedImage);
	}

private:

};
