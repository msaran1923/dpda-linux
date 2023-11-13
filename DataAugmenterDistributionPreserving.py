import cv2
import os
import time
import threading
import numpy as np
from FeatureExtractor import FeatureExtractor
from DensityDecreasingPath import DensityDecreasingPath
from DataAugmenter import DataAugmenter
from NoiseGenerator import NoiseGenerator
from RGB import RGB
from KernelFunctions import EpanechnikovKernel

class DataAugmenterDistributionPreserving(DataAugmenter):
    def __init__(self, imageLoader, imageSaver, noiseGenerator, DPDA_Power):
        super().__init__(imageLoader, imageSaver)
        self.noiseGenerator = noiseGenerator
        self.DPDA_Power = DPDA_Power
        self.lock = threading.Lock()

    def findCoefficients(self, x, dimensionIndex):
        n = len(x)

        A = np.zeros((n, n), dtype=np.float64)
        A.fill(1.0)
        np.fill_diagonal(A, 4.0)

        f = np.zeros((n, 1), dtype=np.float64)

        for i in range(1, n - 1):
            y_ip = x[i - 1][0][dimensionIndex]
            y_i = x[i][0][dimensionIndex]
            y_in = x[i + 1][0][dimensionIndex]

            f[i, 0] = 6.0 * ((y_in - y_i) - (y_i - y_ip))

        s = np.linalg.solve(A, f)

        degree = 3
        abcd = np.zeros((n - 1, degree + 1), dtype=np.float64)

        for i in range(n - 1):
            y_i = x[i][0][dimensionIndex]
            y_in = x[i + 1][0][dimensionIndex]

            abcd[i, 0] = (s[i + 1, 0] - s[i, 0]) / 6.0
            abcd[i, 1] = s[i, 0] / 2.0
            abcd[i, 2] = (y_in - y_i) - (2 * s[i, 0] + s[i + 1, 0]) / 6.0
            abcd[i, 3] = y_i

        return abcd

    def regularizePoints(self, points, labelCount):
        refinedPoints = []

        n = len(points)

        # repeats the point labelCount times
        if n == 1:
            for i in range(labelCount):
                refinedPoints.append(points[0].copy())

        # linearly interpolates between first and last points
        if n == 2:
            pts1 = points[0]
            pts2 = points[1]
            for i in range(labelCount):
                alpha = i / (labelCount - 1.0)
                interpolatedPoint = (1.0 - alpha) * pts1 + alpha * pts2
                refinedPoints.append(interpolatedPoint)

        # cubic spline interpolation between points
        if n >= 3:
            abcd_x = self.findCoefficients(points, 0)
            abcd_y = self.findCoefficients(points, 1)
            abcd_z = self.findCoefficients(points, 2)

            for i in range(labelCount):
                t = i / (labelCount - 1.0) * (n - 1.0)
                t_i = int(t)
                i_d = t - t_i

                if t_i == n - 1 and i_d == 0:
                    refinedPoints.append(points[n - 1])
                    break

                abcd_x_i = abcd_x[t_i]
                abcd_y_i = abcd_y[t_i]
                abcd_z_i = abcd_z[t_i]

                ax_i, bx_i, cx_i, dx_i = abcd_x_i
                ay_i, by_i, cy_i, dy_i = abcd_y_i
                az_i, bz_i, cz_i, dz_i = abcd_z_i

                x_c = ax_i * i_d**3 + bx_i * i_d**2 + cx_i * i_d + dx_i
                y_c = ay_i * i_d**3 + by_i * i_d**2 + cy_i * i_d + dy_i
                z_c = az_i * i_d**3 + bz_i * i_d**2 + cz_i * i_d + dz_i

                interpolatedPoint = np.array([x_c, y_c, z_c], dtype=np.float32)
                refinedPoints.append(interpolatedPoint)

        return refinedPoints

    def isEmpty(self, allPathPointsData, dataCount):
        for i in range(dataCount):
            # 0 - red, 1 - green, 2 - blue
            if allPathPointsData[i][0] > 0 and allPathPointsData[i][1] > 0 and allPathPointsData[i][2] > 0:
                return False
        return True

    def createDensityDecreasingPath(self, image, hInitial, L, features, flann_index, kernelFunctor, d, K, convergenceTolerence, maximumLength):
        allPathPoints = np.zeros((image.shape[0] * image.shape[1], L, 3), dtype=np.float32)

        query = np.empty((1, d), dtype=np.float32)

        direction = -1.0
        h = hInitial

        noPathPointCount = 0

        pixelIndex = 0
        for y in range(image.shape[0]):
            irow = image[y]

            for x in range(image.shape[1]):
                if self.isEmpty(allPathPoints[pixelIndex], L):
                    query[0, 0] = irow[x, 0]
                    query[0, 1] = irow[x, 1]
                    query[0, 2] = irow[x, 2]

                    minimumPointCount = 32
                    similarityDistance = d * 2.0**2  # squared distance
                    similarPixels = []
                    hv = [h]
                    unregularPathPoints = DensityDecreasingPath.findPath(
                        features, flann_index, K, minimumPointCount, hv, query, convergenceTolerence, L // 2,
                        direction, maximumLength, kernelFunctor, similarPixels, similarityDistance, image.shape[1], image.shape[0])
                    h = hv[0]

                    pathPointCount = len(unregularPathPoints)
                    if pathPointCount <= 1:
                        # print(f'P({x}, {y}) = {int(irow[x, 0])} {int(irow[x, 1])} {int(irow[x, 2])}')
                        noPathPointCount += 1

                    pathPoints = self.regularizePoints(unregularPathPoints, L)

                    for i in range(len(pathPoints)):
                        allPathPoints[pixelIndex][i] = pathPoints[i]  # Assign red, green, and blue

                    # For similar pixels, use obtained density-decreasing centers to speed up
                    for pts in similarPixels:
                        pixelIndexSimilar = int(pts[1] * image.shape[1] + pts[0])
                        similarPathPointsData = allPathPoints[pixelIndexSimilar]

                        for i in range(len(pathPoints)):
                            similarPathPointsData[i] = pathPoints[i]  # Assign red, green, and blue

                    h = max(h * 0.99, hInitial)

                pixelIndex += 1
            print(f' [{h:.5f}] ', end='', flush=True)

        print(f'\n{100.0 * noPathPointCount / (image.shape[1] * image.shape[0])}% no density-decrease\n\n')

        return allPathPoints

    def createAugmentedImages(self, image, allPathPoints, d, labelCount, augmentationCount, applyDPDA_Decisions):
        augmentedImages = []

        for i in range(augmentationCount):
            applyDPDA = not allPathPoints is None and applyDPDA_Decisions[i]

            DPDA_Baseline = self.randUnity()
            brightnessBaseline = 1.0 - DPDA_Baseline
            # print('DPDA_Baseline =', DPDA_Baseline, ', brightnessBaseline =', brightnessBaseline)

            perlinRoghness = np.random.uniform(1.0, 5.0)
            perlinNoise = self.noiseGenerator.create(image.shape[1], image.shape[0], perlinRoghness)

            augmentedImage = np.zeros_like(image, dtype=np.uint8)

            pixelIndex = 0
            for y in range(image.shape[0]):
                irow = image[y]
                pmrow = perlinNoise[y]
                airow = augmentedImage[y]

                for x in range(image.shape[1]):
                    rgb = irow[x].copy()

                    if applyDPDA:
                        DPDA_Effect = 1.0 - (pmrow[x] / 255.0)
                        noiseIndexDPDA = min(DPDA_Baseline + DPDA_Effect, 1.0)
                        DPDA_Index = min(int((labelCount - 1) * noiseIndexDPDA), labelCount - 1)
                        DPDA_Data = allPathPoints[pixelIndex]

                        rgb = DPDA_Data[DPDA_Index]

                    airow[x] = rgb
                    pixelIndex += 1

            augmentedImages.append(augmentedImage)

        return augmentedImages

    def distributionPreservingDataAugmentation(self, image, augmentationCount, DPDA_Power, augmentationPercentage):
        # create color features
        d = image.shape[2]
        features = FeatureExtractor.create(image)

        # kd-tree search test
        K = 256
        L = 64
        convergenceTolerance = 0.01 * d
        hInitial = self.estimateH(image)
        maximumLength = DPDA_Power * np.sqrt(hInitial)

        # constructs flann tree (approximate kd-tree search)
        tree_count = 1
        flann_index = cv2.flann_Index(features, {'algorithm': 1, 'trees': tree_count})

        applyDPDA_Decisions = [np.random.randint(0, 100) < augmentationPercentage for _ in range(augmentationCount)]
        atLeastOneDPDA_Application = any(applyDPDA_Decisions)

        kernelFunctor = EpanechnikovKernel()
        allPathPoints = None
        if atLeastOneDPDA_Application:
            allPathPoints = self.createDensityDecreasingPath(image, hInitial, L, features, flann_index,
                                                                 kernelFunctor, d, K, convergenceTolerance, maximumLength)

        augmentedImages = self.createAugmentedImages(image, allPathPoints, d, L, augmentationCount, applyDPDA_Decisions)

        return augmentedImages

    def execute(self, inputDirectory, outputDirectory, imagePath, augmentationCount, scaleFactor, augmentationPercentage):
        fileDirectory = os.path.dirname(imagePath)
        imageFileName = os.path.splitext(os.path.basename(imagePath))[0]
        extension = os.path.splitext(imagePath)[1]

        inputFileName = os.path.join(fileDirectory, imageFileName + extension)

        image = self.imageLoader.load(inputFileName)
        if image is None or image.size == 0:
            print("\n")
            return False

        resizedImage = cv2.resize(image, None, fx=scaleFactor, fy=scaleFactor)

        # process (create augmented images)
        t1 = time.time()

        # DPDA augmentation
        augmentedImages = self.distributionPreservingDataAugmentation(resizedImage, augmentationCount, self.DPDA_Power, augmentationPercentage)

        t2 = time.time()
        duration = t2 - t1

        with self.lock:
            print(f", Elapsed time: {duration:.2f} seconds")

        # write the original image
        outputFileName = os.path.join(fileDirectory, imageFileName + extension)
        outputFileName = outputFileName.replace(inputDirectory, outputDirectory)
        self.imageSaver.save(resizedImage, outputFileName)

        # write augmented images (x count)
        for k, augmentedImage in enumerate(augmentedImages):
            outputFileName = os.path.join(fileDirectory, f"{imageFileName}_{k}{extension}")
            outputFileName = outputFileName.replace(inputDirectory, outputDirectory)

            augmentedImage = self.augmentImage(augmentedImage)
            self.imageSaver.save(augmentedImage, outputFileName)

        return True

    def augmentImage(self, image):
        if self.pipelineDataAugmenter == None:
            return image
        else:
            return self.pipelineDataAugmenter.augmentImage(image)

    def estimateH(self, image):
        distances = []

        max_d = 256 * 256

        for y in range(1, image.shape[0] - 1):
            irow = image[y]
            urow = image[y - 1]
            drow = image[y + 1]

            for x in range(1, image.shape[1] - 1):
                c = irow[x]
                c1 = irow[x - 1]
                c2 = irow[x + 1]
                c3 = urow[x]
                c4 = drow[x]

                d1 = (int(c[0]) - c1[0])**2 + (int(c[1]) - c1[1])**2 + (int(c[2]) - c1[2])**2
                d2 = (int(c[0]) - c2[0])**2 + (int(c[1]) - c2[1])**2 + (int(c[2]) - c2[2])**2
                d3 = (int(c[0]) - c3[0])**2 + (int(c[1]) - c3[1])**2 + (int(c[2]) - c3[2])**2
                d4 = (int(c[0]) - c4[0])**2 + (int(c[1]) - c4[1])**2 + (int(c[2]) - c4[2])**2
                d = min(d1, d2, d3, d4)

                if 1 < d < max_d:
                    distances.append(d)

        median = np.median(distances)

        return max(np.sqrt(median), 1.0)
