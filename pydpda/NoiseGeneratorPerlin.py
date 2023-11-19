import numpy as np
import random
import time
import math
from NoiseGenerator import NoiseGenerator
from PerlinNoise import PerlinNoise

class NoiseGeneratorPerlin(NoiseGenerator):
    runningIndex = 0
        
    def create(self, width, height, roughness):
        perlinNoiseCreator = PerlinNoise(self.getSeed())

        perlinNoise = np.zeros((height, width), dtype=np.uint8)

        noiseScale = self.randRange(0.1, 10.0)
        noiseCenter = self.randRange(0.35, 0.65)

        for y in range(height):
            for x in range(width):
                xf = x / width
                yf = y / height

                noise = perlinNoiseCreator.noise(xf * roughness, yf * roughness, 1.0)
                adjustedNoise = (math.tanh(noiseScale * (noise - noiseCenter)) + 1.0) / 2.0

                perlinNoise[y, x] = int(255 * adjustedNoise)

        return perlinNoise

    def getSeed(self):
        self.runningIndex += 1
        return int(time.time()) + self.runningIndex

    def randRange(self, minValue, maxValue):
        randomRange = 8192
        return (random.randint(0, randomRange - 1) / (randomRange - 1.0)) * (maxValue - minValue) + minValue
