import numpy as np
import random
import time
from NoiseGenerator import NoiseGenerator
from PerlinNoise import PerlinNoise

class NoiseGeneratorPerlin(NoiseGenerator):
    runningIndex = 0

    def create(self, width, height, roughness):
        perlinNoiseCreator = PerlinNoise(self.getSeed())

        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

        perlinNoise = np.fromfunction(lambda i, y, x: self.calculatePixel(
            x, y, width, height, roughness[i], perlinNoiseCreator),
            (len(roughness), height, width),
            dtype=np.uint8)

        return perlinNoise

    def calculatePixel(self, x, y, width, height, roughness,
                       perlinNoiseCreator):
        xf = x / width
        yf = y / height

        noiseScale = np.random.uniform(0.1, 10.0)
        noiseCenter = np.random.uniform(0.35, 0.65)

        noise = perlinNoiseCreator.noise(xf * roughness, yf * roughness, 1.0)
        adjustedNoise = (np.tanh(noiseScale * (noise - noiseCenter)) + 1.0) / 2.0

        return np.vectorize(lambda x: int(255 * x))(adjustedNoise)

    def getSeed(self):
        self.runningIndex += 1
        return int(time.time()) + self.runningIndex

    def randRange(self, minValue, maxValue):
        randomRange = 8192
        return (random.randint(0, randomRange - 1) / (randomRange - 1.0)) * (maxValue - minValue) + minValue
