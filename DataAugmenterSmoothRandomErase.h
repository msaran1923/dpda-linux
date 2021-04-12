#pragma once
#include "OSDefines.h"
#include "DataAugmenter.h"

using namespace std;

class DataAugmenterSmoothRandomErase : public DataAugmenter {
public:
	DataAugmenterSmoothRandomErase(ImageLoader* imageLoader, ImageSaver* imageSaver, NoiseGenerator* noiseGenerator)
		: DataAugmenter(imageLoader, imageSaver)
	{
		this->noiseGenerator = noiseGenerator;
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
			const bool applySmoothRandomErase = (rand() % 100 < augmentationPercentage);

			if (applySmoothRandomErase) {
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

		double perlinRoghness = randRange(1.0, 5.0);
		Mat perlinNoise = noiseGenerator->create(image.cols, image.rows, perlinRoghness);

		const float brightnessBaseline = (float)(1.0 - randUnity());

		for (int y = 0; y < augmentedImage.rows; y++) {
			RGB<uchar>* irow = (RGB<uchar>*)(augmentedImage.data + y * augmentedImage.step);
			unsigned char* pmrow = (unsigned char*)(perlinNoise.data + y * perlinNoise.step);

			for (int x = 0; x < augmentedImage.cols; x++) {
				const float perlinBrightness = (pmrow[x] / 255.0f);			// Perlin noise, per-pixel
				const float noiseIndexBrightness = min(brightnessBaseline + perlinBrightness, 1.0f);
				irow[x].red = brightnessAdjuster(irow[x].red, noiseIndexBrightness);
				irow[x].green = brightnessAdjuster(irow[x].green, noiseIndexBrightness);
				irow[x].blue = brightnessAdjuster(irow[x].blue, noiseIndexBrightness);
			}
		}

		if (pipelineDataAugmenter == 0)
			return augmentedImage;
		else
			return pipelineDataAugmenter->augmentImage(augmentedImage);
	}

private:
	NoiseGenerator* noiseGenerator;

	template<typename T>
	T brightnessAdjuster(T value, float p)
	{
		return (T)max(min((T)(value * p), (T)255), (T)0);
	}

};
