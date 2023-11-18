import cv2
import os
import random
from DataAugmenter import DataAugmenter

class DataAugmenterFlip(DataAugmenter):
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

        # Write the original image
        outputFileName = os.path.join(fileDirectory, imageFileName + extension)
        outputFileName = outputFileName.replace(inputDirectory, outputDirectory)
        self.imageSaver.save(resizedImage, outputFileName)

        # Write augmented images (x count)
        for k in range(augmentationCount):
            applyFlip = random.randint(0, 99) < augmentationPercentage

            if applyFlip:
                outputFileName = f"{fileDirectory}/{imageFileName}_{k}{extension}"
                outputFileName = outputFileName.replace(inputDirectory, outputDirectory)

                augmentedImage = self.augmentImage(resizedImage)
                self.imageSaver.save(augmentedImage, outputFileName)

        return True

    def augmentImage(self, image):
        augmentedImage = image.copy()

        flipCode = random.randint(0, 2)
        augmentedImage = cv2.flip(augmentedImage, flipCode)

        if self.pipelineDataAugmenter is None:
            return augmentedImage
        else:
            return self.pipelineDataAugmenter.augmentImage(augmentedImage)
