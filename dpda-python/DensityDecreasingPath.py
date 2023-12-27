import numpy as np

class DensityDecreasingPath:
    @staticmethod
    def sqr(x):
        return x * x
    
    @staticmethod
    def findPath(features, flann_index, K, Kmin, h, query, convergenceTolerance, maxIteration, direction, maximumLength, kernelFunctor, similarPixels, similarityDistance, imageWidth, imageHeight):
        epsilon = 1e-8
        convergenceTolerance = max(convergenceTolerance, epsilon)

        x = query.copy()
        pathPoints = [x.copy()]

        pixelCount = imageWidth * imageHeight

        mxo = None
        iteration = 0
        while iteration < maxIteration:
            # nearest neighbour search
            unlimitedSearch = -1
            indices, distances = flann_index.knnSearch(x, K, params=unlimitedSearch)
            count = DensityDecreasingPath.findCount(distances, h, Kmin)
            if count == 0:
                break # converged

            indicesArr = indices[0]
            distancesArr = distances[0]

            # perturb farthest neighbour point if it equals to the center point
            minimumRadius = 1.0
            windowRadius = np.sqrt(distancesArr[count - 1])
            if iteration == 0 and windowRadius < minimumRadius:
                DensityDecreasingPath.perturbPoint(x, minimumRadius)
                continue
            h = windowRadius
            
            # to speed-up (find pixels with same/very similar RGB values)
            DensityDecreasingPath.findSimilarFeatures(iteration, count, indicesArr, distancesArr, similarityDistance, imageWidth, pixelCount, similarPixels)

            # calculate mean-shift update
            x_new = DensityDecreasingPath.meanShift(features, kernelFunctor, h, indicesArr, x, count)
            if x_new is None:
                break

            # find mean-shift vector
            mx = x_new - x

            # mean-shift vector convergence check (avoid very small updates, increases efficiency)
            r = np.linalg.norm(mx)
            if r < convergenceTolerance:
                break # converged

            # mean-shift vector length regularization  (projected to domain for constraint 1)
            mx = DensityDecreasingPath.applyLengthRegularization(mx, r, maximumLength)

            if mxo is not None:
                # momentum for gradient descent
                gamma = 0.50
                mx = DensityDecreasingPath.applyMomentumToGradientDescent(mx, mxo, gamma)

                # mean-shift vector direction regularization  (will be projected to domain for constraint 2)					
                maximumAngle = np.pi / 4.0
                mx = DensityDecreasingPath.applySmoothnessRegularization(mx, mxo, maximumAngle)
            mxo = mx

            # move to new point
            x += (direction * mx)

            if DensityDecreasingPath.isOutOfDomain(x):
                break

            # add to the list
            pathPoints.append(x.copy())

            iteration += 1
        
        return pathPoints, h

    @staticmethod
    def findCount(distances, h, minimumPointCount):
        squaredRadius = h**2
        mask = distances[0, minimumPointCount - 1:] >= squaredRadius
        first_index = np.argmax(mask)

        return minimumPointCount + first_index if np.any(mask) else minimumPointCount

    @staticmethod
    def perturbPoint(x, noiseLevel):
        noise = noiseLevel * (np.random.randint(0, 2, size=x.shape[1]) * 2 - 1) * (np.random.rand(1, x.shape[1]) + 1.0)
        x[0, :] = np.clip(x[0, :] + noise, 0.0, 255.0)

    @staticmethod
    def findSimilarFeatures(iteration, count, indicesArr, distancesArr, similarityDistance, imageWidth, pixelCount, similarPixels):
        firstIteration = 0

        if iteration == firstIteration:
            similarPixels.clear()
            mask = distancesArr[:count] <= similarityDistance
            rValues = indicesArr[:count][mask]

            xxValues = rValues % imageWidth
            yyValues = rValues // imageWidth
            
            validIndices = rValues < pixelCount

            similarPixels.extend(zip(xxValues[validIndices], yyValues[validIndices]))

    @staticmethod
    def meanShift(features, kernelFunctor, h, indicesArr, x, count):
        epsilon = 1e-8

        x_i = features[indicesArr[:count], :]
        d = np.linalg.norm((x - x_i) / h, axis=1)
        x_d = kernelFunctor.eval(d)
        
        sqr_x_d = x_d**2

        kde = np.sum(sqr_x_d)
        C_numerator = np.sum(x_i * sqr_x_d.reshape(-1, 1), axis=0)

        if kde < epsilon or np.linalg.norm(C_numerator) < epsilon:
            return None  # converged

        x_new = C_numerator / kde
        x_new = x_new.astype(np.float32)

        return x_new


    @staticmethod
    def applyLengthRegularization(mx, r, maximumLength):
        return min(r, maximumLength) * (mx / r)

    @staticmethod
    def applyMomentumToGradientDescent(mx, mxo, gamma):
        return gamma * mxo + (1.0 - gamma) * mx

    @staticmethod
    def applySmoothnessRegularization(mx, mxo, alpha):
        epsilon = 1e-8

        sOld3D = np.copy(mxo)
        sNew3D = np.copy(mx)

        sOld3DUnit = (1.0 / np.linalg.norm(sOld3D)) * sOld3D
        sNew3DUnit = (1.0 / np.linalg.norm(sNew3D)) * sNew3D

        isProjectionNeeded = np.dot(sOld3DUnit.flatten(), sNew3DUnit.flatten()) < np.cos(alpha)
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
                sOld = v1_hat[0, 0:2].copy()
                sNew = v2_hat[0, 0:2].copy()

                sOldUnit = sOld / np.linalg.norm(sOld)
                sNewUnit = sNew / np.linalg.norm(sNew)

                theta = np.arccos(np.dot(sOldUnit, sNewUnit))
                beta = alpha - theta
                sNewReg1 = DensityDecreasingPath.getRotationMatrix(beta).dot(sNew.transpose()).transpose()
                sNewReg2 = DensityDecreasingPath.getRotationMatrix(-beta).dot(sNew.transpose()).transpose()

                sNewReg = sNewReg1 if np.dot(sNewReg1, sOld) > np.dot(sNewReg2, sOld) else sNewReg2

                if noRotationNeeded:
                    mx[0, 0:2] = sNewReg[0:2]
                    mx[0, 2] = 0.0
                else:
                    sNewProjected = np.zeros((1, 3), dtype=np.float32)
                    sNewProjected[0, 0:2] = sNewReg[0:2]

                    mx = np.dot(np.linalg.inv(R), sNewProjected.transpose()).transpose()
        return mx

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
        return np.dot(a, b)

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
