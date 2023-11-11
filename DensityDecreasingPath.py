import numpy as np
import pyflann
import random

class DensityDecreasingPath:
    @staticmethod
    def sqr(x):
        return x * x
    
    @staticmethod
    def findPath(features, flann_index, K, Kmin, h, query, convergenceTolerance, maxIteration, direction, maximumLength, kernelFunctor, similarPixels, similarityDistance, imageWidth, imageHeight):
        pathPoints = []

        epsilon = 1e-8
        convergenceTolerance = max(convergenceTolerance, epsilon)

        x = query.copy()
        pathPoints.append(x.copy())

        pixelCount = imageWidth * imageHeight

        mxo = None
        iteration = 0
        while iteration < maxIteration:
            # nearest neighbour search
            unlimitedSearch = -1
            indices, distances = flann_index.knnSearch(x, K, params=unlimitedSearch)
            count = DensityDecreasingPath.findCount(distances, h[0], Kmin)
            if count == 0:
                break # converged

            indicesArr = indices[0]
            distancesArr = distances

            # perturb farthest neighbour point if it equals to the center point
            minimumRadius = 1.0
            windowRadius = np.sqrt(distancesArr[0][count - 1])
            if iteration == 0 and windowRadius < minimumRadius:
                DensityDecreasingPath.perturbPoint(x, minimumRadius)
                continue
            h[0] = windowRadius
            
            # to speed-up (find pixels with same/very similar RGB values)
            DensityDecreasingPath.findSimilarFeatures(iteration, count, indicesArr, distancesArr, similarityDistance, imageWidth, pixelCount, similarPixels)

            # calculate mean-shift update
            x_new = DensityDecreasingPath.meanShift(features, kernelFunctor, h[0], indicesArr, x, count)
            if x_new is None:
                break

            # find mean-shift vector
            mx = x_new - x

            # mean-shift vector convergence check (avoid very small updates, increases efficiency)
            r = np.linalg.norm(mx)
            if r < convergenceTolerance:
                break # converged

            # mean-shift vector length regularization  (projected to domain for constraint 1)
            DensityDecreasingPath.applyLengthRegularization(mx, r, maximumLength)

            if mxo is not None:
                # momentum for gradient descent
                gamma = 0.50
                DensityDecreasingPath.applyMomentumToGradientDescent(mx, mxo, gamma)

                # mean-shift vector direction regularization  (will be projected to domain for constraint 2)					
                maximumAngle = np.pi / 4.0
                DensityDecreasingPath.applySmoothnessRegularization(mx, mxo, maximumAngle)
            mxo = mx

            # move to new point
            x += (direction * mx)

            if DensityDecreasingPath.isOutOfDomain(x):
                break

            # add to the list
            pathPoints.append(x.copy())

            iteration += 1
        
        return pathPoints

    @staticmethod
    def findCount(distances, h, minimumPointCount):
        squaredRadius = h * h

        for i in range(minimumPointCount - 1, distances.shape[1]):
            if distances[0, i] >= squaredRadius:
                return i + 1

        return minimumPointCount

    @staticmethod
    def perturbPoint(x, noiseLevel):
        for i in range(x.shape[1]):
            noise = noiseLevel * (random.randint(0, 1) * 2 - 1) * (random.random() + 1.0)
            x[0, i] = max(min(x[0, i] + noise, 255.0), 0.0)

    @staticmethod
    def findSimilarFeatures(iteration, count, indicesArr, distancesArr, similarityDistance, imageWidth, pixelCount, similarPixels):
        first_iteration = 0
        if iteration == first_iteration:
            similarPixels.clear()

            for i in range(count):
                if distancesArr[0][i] <= similarityDistance:
                    r = indicesArr[i]
                    if r < pixelCount:
                        xx = r % imageWidth
                        yy = r // imageWidth

                        similarPixels.append((xx, yy))

    @staticmethod
    def meanShift(features, kernelFunctor, h, indicesArr, x, count):
        epsilon = 1e-8

        kde = 0.0
        C_numerator = np.zeros_like(x, dtype=np.float64)

        for i in range(count):
            r = indicesArr[i]
            x_i = features[r:r+1, :]

            d = np.linalg.norm((x - x_i) / h)
            x_d = kernelFunctor.eval(d)

            sqr_x_d = x_d * x_d

            kde += sqr_x_d
            C_numerator += x_i * sqr_x_d

        if kde < epsilon or np.linalg.norm(C_numerator) < epsilon:
            return None  # converged

        x_new = C_numerator / kde
        x_new = x_new.astype(np.float32)

        return x_new


    @staticmethod
    def applyLengthRegularization(mx, r, maximumLength):
        mx = min(r, maximumLength) * (mx / r)

    @staticmethod
    def applyMomentumToGradientDescent(mx, mxo, gamma):
        mx = gamma * mxo + (1.0 - gamma) * mx

    @staticmethod
    def applySmoothnessRegularization(mx, mxo, alpha):
        epsilon = 1e-8

        sOld3D = mxo[0].copy()
        sNew3D = mx[0].copy()

        sOld3DUnit = sOld3D / np.linalg.norm(sOld3D)
        sNew3DUnit = sNew3D / np.linalg.norm(sNew3D)

        isProjectionNeeded = np.dot(sOld3DUnit, sNew3DUnit) < np.cos(alpha)
        if isProjectionNeeded:
            a = np.cross(sOld3DUnit, sNew3DUnit)
            isInSameDirection = np.linalg.norm(a) < epsilon

            if not isInSameDirection:
                R = DensityDecreasingPath.calculateVectorAlignmentRotationMatrix(a, epsilon)
                noRotationNeeded = R is None

                if noRotationNeeded:
                    v1_hat = sOld3D
                    v2_hat = sNew3D
                else:
                    v1_hat = R.dot(sOld3D.transpose()).transpose()
                    v2_hat = R.dot(sNew3D.transpose()).transpose()

                # now we are working in 2D (R projects vectors into xy-plane)
                sOld = v1_hat[0:2].copy()
                sNew = v2_hat[0:2].copy()

                sOldUnit = sOld / np.linalg.norm(sOld)
                sNewUnit = sNew / np.linalg.norm(sNew)

                theta = np.arccos(np.dot(sOldUnit, sNewUnit))
                beta = alpha - theta
                sNewReg1 = DensityDecreasingPath.getRotationMatrix(beta).dot(sNew.transpose()).transpose()
                sNewReg2 = DensityDecreasingPath.getRotationMatrix(-beta).dot(sNew.transpose()).transpose()

                sNewReg = sNewReg1 if np.dot(sNewReg1, sOld) > np.dot(sNewReg2, sOld) else sNewReg2

                if noRotationNeeded:
                    mx[0, 0:2] = sNewReg[0, 0:2]
                    mx[0, 2] = 0.0
                else:
                    sNewProjected = np.zeros((1, 3), dtype=np.float32)
                    sNewProjected[0, 0:2] = sNewReg[0:2]

                    mx = np.dot(np.linalg.inv(R), sNewProjected.transpose()).transpose()

    # Efficiently Building a Matrix to Rotate One Vector to Another
	# https://www.tandfonline.com/doi/abs/10.1080/10867651.1999.10487509
    @staticmethod
    def calculateVectorAlignmentRotationMatrix(f, epsilon):
        f = (1.0 / np.linalg.norm(f)) * f
        t = np.zeros((1, 3), dtype=np.float32)
        t[0, 2] = 1.0

        v = np.cross(f, t)
        s = np.linalg.norm(v)

        noRotationNeeded = abs(s) < epsilon
        if noRotationNeeded:
            return None

        u = v / s
        c = np.dot(f, t[0])
        if abs(c) >= 1 - epsilon:
            return None

        h = (1 - c) / (1 - c * c)

        vx, vy, vz = v[0]
        R = np.eye(3, dtype=np.float32)

        R[0, 0] = c + h * vx * vx
        R[0, 1] = h * vx * vy - vz
        R[0, 2] = h * vx * vz + vy

        R[1, 0] = h * vx * vy + vz
        R[1, 1] = c + h * vy * vy
        R[1, 2] = h * vy * vz - vx

        R[2, 0] = h * vx * vz - vy
        R[2, 1] = h * vy * vz + vx
        R[2, 2] = c + h * vz * vz

        return R

    @staticmethod
    def dotProduct(a, b):
        sum = 0.0
        for i in range(a.size):
            sum += a[i] * b[i]

        return sum

    @staticmethod
    def getRotationMatrix(beta):
        cosBeta = np.cos(beta)
        sinBeta = np.sin(beta)

        R = np.array([[cosBeta, -sinBeta],
                      [sinBeta, cosBeta]], dtype=np.float32)

        return R

    @staticmethod
    def isOutOfDomain(x):
        return (x < 0.0).any() or (x > 255.0).any()
