import cv2
import os
import random
from DataAugmenter import DataAugmenter

class DataAugmenterHistogramEqualization(DataAugmenter):
    def __init__(self, ImageLoader, imageSaver):
        super().__init__(ImageLoader, imageSaver)

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
            applyHistogramEqualization = random.randint(0, 99) < augmentationPercentage

            if applyHistogramEqualization:
                outputFileName = f"{fileDirectory}/{imageFileName}_{k}{extension}"
                outputFileName = outputFileName.replace(inputDirectory, outputDirectory)

                augmentedImage = self.augmentImage(resizedImage)
                self.imageSaver.save(augmentedImage, outputFileName)

        return True

    def augmentImage(self, image):
        augmentedImage = image.copy()

        if image.shape[2] == 3:
            rgbChannels = cv2.split(image)

            equalizedImages = []
            for i in range(3):
                equalizedImage = cv2.equalizeHist(rgbChannels[i])
                equalizedImages.append(equalizedImage)
            augmentedImage = cv2.merge(equalizedImages)
        if image.shape[2] == 1:
            augmentedImage = cv2.equalizeHist(image)

        if self.pipelineDataAugmenter is None:
            return augmentedImage
        else:
            return self.pipelineDataAugmenter.augmentImage(augmentedImage)
