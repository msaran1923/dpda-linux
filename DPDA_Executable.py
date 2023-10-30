#!/usr/bin/env python3

import time
from FileRepositoryDirectoryBased import FileRepositoryDirectoryBased
from ImageLoaderOpenCV import ImageLoaderOpenCV
from ImageSaverOpenCV import ImageSaverOpenCV
from NoiseGeneratorPerlin import NoiseGeneratorPerlin

print('Distribution-preserving Data Augmentation (DPDA) v0.02p')

directoryNames = []
directoryNames.append('train')
# add directories as you wish
#directoryNames.append('val')
#directoryNames.append('test')

inputDirectory = 'build/images'
outputDirectory = 'build/results'

fileRepository = FileRepositoryDirectoryBased(inputDirectory, outputDirectory, directoryNames)
imagePaths = fileRepository.getImagePaths()

t1 = time.time()

imageLoader = ImageLoaderOpenCV()
imageSaver = ImageSaverOpenCV()
noiseGenerator = NoiseGeneratorPerlin()

# to-do: implement the rest of the code

t2 = time.time()
duration = t2 - t1
print('Elapsed time:', duration, 'seconds')

logFileName = '_imageWithProblems.txt'
imageLoader.saveUnreadedImages(logFileName)
