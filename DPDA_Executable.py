#!/usr/bin/env python3

from FileRepositoryDirectoryBased import FileRepositoryDirectoryBased

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
