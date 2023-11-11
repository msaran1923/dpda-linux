#!/usr/bin/env python3

import time
from FileRepositoryDirectoryBased import FileRepositoryDirectoryBased
from DataAugmenterDistributionPreserving import DataAugmenterDistributionPreserving
from DataAugmenterRandomErase import DataAugmenterRandomErase
from DataAugmenterGammaCorrection import DataAugmenterGammaCorrection
from DataAugmenterHistogramEqualization import DataAugmenterHistogramEqualization
from DataAugmenterFlip import DataAugmenterFlip
from NoiseGeneratorPerlin import NoiseGeneratorPerlin
from ImageLoaderOpenCV import ImageLoaderOpenCV
from ImageSaverOpenCV import ImageSaverOpenCV

print('Distribution-preserving Data Augmentation (DPDA) v0.03p')

directoryNames = []
directoryNames.append('train')
# add directories as you wish
#directoryNames.append('val')
#directoryNames.append('test')

inputDirectory = 'build/images'
outputDirectory = 'build/results'

fileRepository = FileRepositoryDirectoryBased(inputDirectory, outputDirectory, directoryNames)
imagePaths = fileRepository.getImagePaths()

# augment images - begin
t1 = time.time()

imageLoader = ImageLoaderOpenCV()
imageSaver = ImageSaverOpenCV()
noiseGenerator = NoiseGeneratorPerlin()
DPDA_Power = 1.0
dataAugmenterDistributionPreserving = DataAugmenterDistributionPreserving(imageLoader, imageSaver, noiseGenerator, DPDA_Power)
dataAugmenterRandomErase = DataAugmenterRandomErase(imageLoader, imageSaver)
dataAugmenterHistogramEqualization = DataAugmenterHistogramEqualization(imageLoader, imageSaver)
dataAugmenterGammaCorrection = DataAugmenterGammaCorrection(imageLoader, imageSaver)
dataAugmenterFlip = DataAugmenterFlip(imageLoader, imageSaver)

# Mix augmentation methods by adding to pipeline
# InputImage --> DataAugmenterFlip --> DataAugmenterGammaCorrection  -->  DataAugmenterRandomErase --> AugmentedImage
# First, create a dataAugmenter with the first augmentation method you wish
# Then, add another augmentation methos to the pipeline
# e.g. To combine Flip, GamaCorrectin, and Random erase you should write:
# DataAugmenter& dataAugmenter = dataAugmenterFlip;
# dataAugmenter.setPipelineDataAugmenter(&dataAugmenterGammaCorrection);
# dataAugmenterGammaCorrection.setPipelineDataAugmenter(&dataAugmenterRandomErase);

# To augment images only with Distribution Preserving Data Augmentation (DPDA), use this statement
dataAugmenter = dataAugmenterDistributionPreserving
# Uncomment the pipeline statements if you want to mix up augmentation methods
# dataAugmenter.setPipelineDataAugmenter(&dataAugmenterFlip);
# dataAugmenter.setPipelineDataAugmenter(&dataAugmenterRandomErase);
# dataAugmenter.setPipelineDataAugmenter(&dataAugmenterGammaCorrection);

for i in range(len(imagePaths)):
    filePath = imagePaths[i]
    augmentationCount = 5
    scaleFactor = 1.0
    augmentationPercentage = 100

    dataAugmenter.execute(inputDirectory, outputDirectory, imagePaths[i], augmentationCount, scaleFactor, augmentationPercentage)

t2 = time.time()
duration = t2 - t1
print('Elapsed time:', duration, 'seconds')
# augment images - end

logFileName = '_imageWithProblems.txt'
imageLoader.saveUnreadedImages(logFileName)
