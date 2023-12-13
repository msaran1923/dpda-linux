import cv2
from ImageSaver import ImageSaver

class ImageSaverOpenCV(ImageSaver):
    def save(self, image, outputFileName):
        cv2.imwrite(outputFileName, image)
