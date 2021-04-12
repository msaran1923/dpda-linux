#pragma once
#include "OSDefines.h"
#include "DataAugmenter.h"


class DataAugmentationPipeline {
public:	
	
	bool execute(String inputDirectory, String outputDirectory, path& imagePath,
		int augmentationCount, double scaleFactor, int percentage)
	{
		// DataAugmentationPipeline augmentationPipeline;
		// augmentationPipeline.add(new DataAugmenterDistributionPreserving(), 100);
		// augmentationPipeline.add(new DataAugmenterSmoothRandomErase(), 50);
		// augmentationPipeline.add(new DataAugmenterRandomErase(), 100);
	}

private:
	vector<DataAugmenter*> dataAugmenters;

};
