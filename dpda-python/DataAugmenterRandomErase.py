import cv2
import os
import random
from DataAugmenter import DataAugmenter

class DataAugmenterRandomErase(DataAugmenter):
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
            applyRandomErase = random.randint(0, 99) < augmentationPercentage

            if applyRandomErase:
                outputFileName = f"{fileDirectory}/{imageFileName}_{k}{extension}"
                outputFileName = outputFileName.replace(inputDirectory, outputDirectory)

                augmentedImage = self.augmentImage(resizedImage)
                self.imageSaver.save(augmentedImage, outputFileName)

        return True

    def augmentImage(self, image):
        augmentedImage = image.copy()

        minPatchWidth = int(0.05 * image.shape[1])
        minPatchHeight = int(0.05 * image.shape[0])
        maxPatchWidth = int(0.50 * image.shape[1])
        maxPatchHeight = int(0.50 * image.shape[0])

        randomWidth = max(random.randint(0, maxPatchWidth), minPatchWidth)
        randomHeight = max(random.randint(0, maxPatchHeight), minPatchHeight)

        x1 = max(random.randint(0, image.shape[1] - minPatchWidth), 1)
        y1 = max(random.randint(0, image.shape[0] - minPatchHeight), 1)
        x2 = min(x1 + randomWidth, image.shape[1] - 1)
        y2 = min(y1 + randomHeight, image.shape[0] - 1)

        augmentedImage[y1:y2, x1:x2] = [0, 0, 0]

        if self.pipelineDataAugmenter is None:
            return augmentedImage
        else:
            return self.pipelineDataAugmenter.augmentImage(augmentedImage)
