import cv2
import numpy as np

class FeatureExtractor:
    @staticmethod
    def create(image):
        # Create pyramid
        pyramidImages = []

        pyramidLevelCount = 4  # 1
        n = 0
        for k in range(pyramidLevelCount):
            scaleFactor = 2 ** (-k)

            if k == 0:
                pyramidImage = image.copy()
            else:
                pyramidImage = cv2.resize(image, None, fx=scaleFactor, fy=scaleFactor, interpolation=cv2.INTER_LANCZOS4)

            pyramidImages.append(pyramidImage)

            n += pyramidImage.shape[0] * pyramidImage.shape[1]

        # create features using images in the pyramid
        features = np.empty((n, image.shape[2]), dtype=np.float32)

        index = 0
        for pyramidImage in pyramidImages:
            for y in range(pyramidImage.shape[0]):
                for x in range(pyramidImage.shape[1]):
                    features[index] = pyramidImage[y, x]  # assigns red, blue and green

                    index += 1

        return features
