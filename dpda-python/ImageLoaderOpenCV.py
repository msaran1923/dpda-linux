import cv2
from ImageLoader import ImageLoader

class ImageLoaderOpenCV(ImageLoader):
    def load(self, imagePath):
        inputFileName = self.getFileName(imagePath)
        image = cv2.imread(str(inputFileName), cv2.IMREAD_COLOR)

        if image is None:
            self.addImageLoadError(imagePath)

        return image
