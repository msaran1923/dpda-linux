#pragma once
#include "OSDefines.h"
#include "ImageLoader.h"
#include "ImageSaver.h"


class DataAugmenter {
public:
	DataAugmenter(ImageLoader* imageLoader, ImageSaver* imageSaver)
	{
		this->imageLoader = imageLoader;
		this->imageSaver = imageSaver;
		this->pipelineDataAugmenter = 0;

		srand((int)time(0));
	}

	virtual bool execute(String inputDirectory, String outputDirectory, path& imagePath,
						 int augmentationCount, double scaleFactor, int augmentationPercentage) = 0;

	virtual Mat augmentImage(Mat image) = 0;

	void setPipelineDataAugmenter(DataAugmenter* pipelineDataAugmenter) {
		this->pipelineDataAugmenter = pipelineDataAugmenter;
	}

	template<typename T>
	inline T sqr(T x) {
		return (x * x);
	}

	template<typename T>
	inline static T cube(T x) {
		return (x * x * x);
	}

	inline static double randUnity()
	{
		const int randomRange = 8192;
		return (rand() % randomRange) / (randomRange - 1.0);
	}

	inline static double randRange(double minValue, double maxValue)
	{
		const int randomRange = 8192;
		return (rand() % randomRange) / (randomRange - 1.0) * (maxValue - minValue) + minValue;
	}

protected:
	ImageLoader* imageLoader;
	ImageSaver* imageSaver;
	DataAugmenter* pipelineDataAugmenter;

};
