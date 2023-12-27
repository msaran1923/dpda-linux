import cv2
import numpy as np

class FeatureExtractor:
    @staticmethod
    def create(image):
        # Create pyramid
        pyramidLevelCount = 4
        pyramidImages = [image] + [cv2.resize(image, None, fx=2 ** (-k), fy=2 ** (-k), interpolation=cv2.INTER_LANCZOS4) for k in range(1, pyramidLevelCount)]

        # Flatten pyramid images into features
        features = np.concatenate([pyramidImage.reshape(-1, image.shape[2]) for pyramidImage in pyramidImages])

        return features.astype(np.float32)
