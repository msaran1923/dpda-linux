#!/usr/bin/env python3

import time
from FileRepositoryDirectoryBased import FileRepositoryDirectoryBased
from DataAugmentationPipeline import DataAugmentationPipeline
from DataAugmenterDistributionPreserving import DataAugmenterDistributionPreserving
from DataAugmenterRandomErase import DataAugmenterRandomErase
from DataAugmenterGammaCorrection import DataAugmenterGammaCorrection
from DataAugmenterHistogramEqualization import DataAugmenterHistogramEqualization
from DataAugmenterFlip import DataAugmenterFlip
from ImageLoaderOpenCV import ImageLoaderOpenCV
from ImageSaverOpenCV import ImageSaverOpenCV

print('Distribution-preserving Data Augmentation (DPDA) v1.07p')

directoryNames = []
directoryNames.append('train')
# add directories as you wish
#directoryNames.append('val')
#directoryNames.append('test')

inputDirectory = 'images'
outputDirectory = 'results'

fileRepository = FileRepositoryDirectoryBased(inputDirectory, outputDirectory, directoryNames)
imagePaths = fileRepository.getImagePaths()

# augment images - begin
t1 = time.time()

imageLoader = ImageLoaderOpenCV()
imageSaver = ImageSaverOpenCV()
dataAugmenter = DataAugmentationPipeline()
dataAugmenterRandomErase = DataAugmenterRandomErase(imageLoader, imageSaver)
dataAugmenterHistogramEqualization = DataAugmenterHistogramEqualization(imageLoader, imageSaver)
dataAugmenterGammaCorrection = DataAugmenterGammaCorrection(imageLoader, imageSaver)
dataAugmenterFlip = DataAugmenterFlip(imageLoader, imageSaver)

# Uncomment the pipeline statements if you want to mix up augmentation methods
#dataAugmenter.appendToPipeline(dataAugmenterFlip)
#dataAugmenter.appendToPipeline(dataAugmenterGammaCorrection)
#dataAugmenter.appendToPipeline(dataAugmenterHistogramEqualization)
#dataAugmenter.appendToPipeline(dataAugmenterRandomErase)

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
