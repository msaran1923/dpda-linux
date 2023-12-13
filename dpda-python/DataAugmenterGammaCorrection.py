import cv2
import os
import random
from DataAugmenter import DataAugmenter
import numpy as np

class DataAugmenterGammaCorrection(DataAugmenter):
    def __init__(self, imageLoader, imageSaver):
        super().__init__(imageLoader, imageSaver)

    def execute(self, inputDirectory, outputDirectory, imagePath, augmentationCount, scaleFactor, augmentationPercentage):
        fileDirectory = os.path.dirname(imagePath)
        imageFileName, extension = os.path.splitext(os.path.basename(imagePath))

        inputFileName = os.path.join(fileDirectory, imageFileName + extension)

        image = self.imageLoader.load(inputFileName)
        if image is None:
            print()
            return False

        resizedImage = cv2.resize(image, None, fx=scaleFactor, fy=scaleFactor)

        # write original image
        outputFileName = os.path.join(fileDirectory, imageFileName + extension)
        outputFileName = outputFileName.replace(inputDirectory, outputDirectory)
        self.imageSaver.save(resizedImage, outputFileName)

        # write augmented images (x count)
        for k in range(augmentationCount):
            applyGammaCorrection = random.randint(0, 99) < augmentationPercentage

            if applyGammaCorrection:
                outputFileName = f"{fileDirectory}/{imageFileName}_{k}{extension}"
                outputFileName = outputFileName.replace(inputDirectory, outputDirectory)

                augmentedImage = self.augmentImage(resizedImage)
                self.imageSaver.save(augmentedImage, outputFileName)

        return True

    def augmentImage(self, image):
        augmentedImage = image.copy()

        gamma = random.uniform(0.001, 2.5)
        invGamma = 1.0 / gamma

        lookUpTable = np.array([pow(i / 255.0, gamma) * 255 for i in range(256)]).astype(np.uint8)
        augmentedImage = cv2.LUT(image, lookUpTable)

        if self.pipelineDataAugmenter is None:
            return augmentedImage
        else:
            return self.pipelineDataAugmenter.augmentImage(augmentedImage)
