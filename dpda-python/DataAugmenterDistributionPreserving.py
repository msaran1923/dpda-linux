import cv2
import os
import time
import threading
import h5py
import numpy as np
from FeatureExtractor import FeatureExtractor
from DensityDecreasingPath import DensityDecreasingPath
from DataAugmenter import DataAugmenter
from KernelFunctions import EpanechnikovKernel

class DataAugmenterDistributionPreserving(DataAugmenter):
    def __init__(self, imageLoader, imageSaver, noiseGenerator, DPDA_Power):
        super().__init__(imageLoader, imageSaver)
        self.noiseGenerator = noiseGenerator
        self.DPDA_Power = DPDA_Power
        self.lock = threading.Lock()

    def findCoefficients(self, x, dimensionIndex):
        n = len(x)

        y = np.array([point[0][dimensionIndex] for point in x], dtype=np.float64)

        A = np.eye(n) * 4.0
        np.fill_diagonal(A[1:], 1.0)
        np.fill_diagonal(A[:, 1:], 1.0)

        f = np.zeros((n, 1), dtype=np.float64)
        f[1:-1, 0] = 6.0 * (y[2:] - 2 * y[1:-1] + y[:-2])

        s = np.linalg.solve(A, f)

        degree = 3
        abcd = np.zeros((n - 1, degree + 1), dtype=np.float64)

        abcd[:, 0] = (s[1:, 0] - s[:-1, 0]) / 6.0
        abcd[:, 1] = s[:-1, 0] / 2.0
        abcd[:, 2] = (y[1:] - y[:-1]) - (2 * s[:-1, 0] + s[1:, 0]) / 6.0
        abcd[:, 3] = y[:-1]

        return abcd

    def regularizePoints(self, points, labelCount):
        refinedPoints = []

        n = len(points)

        # repeats the point labelCount times
        if n == 1:
            refinedPoints = [points[0][0].copy()] * labelCount

        # linearly interpolates between first and last points
        if n == 2:
            pts1 = points[0][0]
            pts2 = points[1][0]
            alpha_values = np.linspace(0, 1, labelCount)
            refinedPoints = [(1 - alpha) * pts1 + alpha * pts2 for alpha in alpha_values]

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
                    refinedPoints.append(points[n - 1][0])
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
        check = np.all(allPathPointsData[:dataCount, :3] > 0, axis=1)
        return not np.any(check)

    def createDensityDecreasingPath(self, image, hInitial, L, features, flann_index, kernelFunctor, d, K, convergenceTolerence, maximumLength):
        direction = -1.0
        h = hInitial
        similarityDistance = d * 4.0
        noPathPointCount, minimumPointCount = 0, 32
        allPathPoints = np.zeros((image.shape[0] * image.shape[1], L, 3), dtype=np.float32)
        query = np.expand_dims(np.array(image), axis=2).astype(np.float32)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                pixelIndex = y * image.shape[1] + x
                if self.isEmpty(allPathPoints[pixelIndex], L):
                    similarPixels = []

                    unregularPathPoints, h = DensityDecreasingPath.findPath(
                        features, flann_index, K, minimumPointCount, h, query[y, x], convergenceTolerence, L // 2,
                        direction, maximumLength, kernelFunctor, similarPixels, similarityDistance, image.shape[1], image.shape[0])

                    noPathPointCount += (len(unregularPathPoints) <= 1)
                    pathPoints = self.regularizePoints(unregularPathPoints, L)
                    allPathPoints[pixelIndex] = pathPoints

                    # For similar pixels, use obtained density-decreasing centers to speed up
                    similarPixelsArray = np.array(similarPixels)
                    pixelIndices = np.round(similarPixelsArray[:, 1] * image.shape[1] + similarPixelsArray[:, 0]).astype(int)
                    allPathPoints[pixelIndices, :len(pathPoints)] = pathPoints

                    h = max(h * 0.99, hInitial)
            print(f' [{h:.5f}] ', end='', flush=True)

        print(f'\n{100.0 * noPathPointCount / (image.shape[1] * image.shape[0])}% no density-decrease\n\n')

        return allPathPoints

    def createAugmentedImages(self, image, allPathPoints, d, labelCount, augmentationCount, applyDPDA_Decisions):
        augmentedImages = []
        DPDA_Baselines = self.randUnity(augmentationCount)
        perlinRoughness = np.random.uniform(1.0, 5.0, size=augmentationCount)
        perlinNoises = self.noiseGenerator.create(image.shape[1], image.shape[0], perlinRoughness)

        for i in range(augmentationCount):
            augmentedImage = image.copy()

            applyDPDA = not allPathPoints is None and applyDPDA_Decisions[i]
            if applyDPDA:
                DPDA_Effects = 1.0 - (perlinNoises[i] / 255.0)
                noiseIndexDPDA = np.minimum(DPDA_Baselines[i] + DPDA_Effects, 1.0)
                DPDA_Indices = np.minimum(np.floor((labelCount - 1) * noiseIndexDPDA), labelCount - 1).astype(int)

                yIndices, xIndices = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
                pixelIndices = yIndices * image.shape[1] + xIndices
                rgbValues = allPathPoints[pixelIndices.flatten(), DPDA_Indices.flatten()]

                augmentedImage = np.reshape(rgbValues, augmentedImage.shape)

            augmentedImages.append(augmentedImage.astype(np.uint8))

        return augmentedImages

    def distributionPreservingDataAugmentation(self, image, augmentationCount, DPDA_Power, augmentationPercentage, imageFileName):
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
            if os.path.exists(f'pathpoints/pathpoint_{imageFileName}.hdf5'):
                allPathPoints = self.allPathPointsLoader(imageFileName)
            else:
                allPathPoints = self.createDensityDecreasingPath(image, hInitial, L, features, flann_index,
                                                                 kernelFunctor, d, K, convergenceTolerance, maximumLength)
                self.allPathPointsWriter(allPathPoints, imageFileName)

        augmentedImages = self.createAugmentedImages(image, allPathPoints, d, L, augmentationCount, applyDPDA_Decisions)

        return augmentedImages

    def allPathPointsWriter(self, allPathPoints, imageFileName):
        fileName = f'pathpoints/pathpoint_{imageFileName}.hdf5'
        datasetName = 'allPathPoints'

        with h5py.File(fileName, 'w') as h5io:
            h5io.create_dataset(datasetName, data=allPathPoints)
        

    def allPathPointsLoader(self, imageFileName):   
        fileName = f'pathpoints/pathpoint_{imageFileName}.hdf5'
        with h5py.File(fileName, 'r') as f:
            allPathPoints = np.array(f['allPathPoints'])
        return allPathPoints

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
        augmentedImages = self.distributionPreservingDataAugmentation(resizedImage, augmentationCount, self.DPDA_Power, augmentationPercentage, imageFileName)

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
        irow = image[1:-1, 1:-1]
        urow = image[:-2, 1:-1]
        drow = image[2:, 1:-1]

        c = irow
        c1 = image[1:-1, :-2]
        c2 = image[1:-1, 2:]
        c3 = urow
        c4 = drow

        d1 = np.sum((c - c1)**2, axis=-1)
        d2 = np.sum((c - c2)**2, axis=-1)
        d3 = np.sum((c - c3)**2, axis=-1)
        d4 = np.sum((c - c4)**2, axis=-1)

        distances = np.minimum.reduce([d1, d2, d3, d4])
        valid_distances = distances[(1 < distances) & (distances < 256 * 256)]
        median = np.median(valid_distances)

        return max(np.sqrt(median), 1.0)
