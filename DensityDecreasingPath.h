#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "KernelFunctions.h"

using namespace cv;


class DensityDecreasingPath {
public:
	template<typename T>
	static inline T sqr(T x) {
		return (x * x);
	}

	static vector<Mat> findPath(Mat& features, flann::Index& flann_index, int K, int Kmin, float& h,
								InputArray query, float convergenceTolerance, int maxIteration,
								float direction, float maximumLength, KernelFunctor& kernelFunctor,
								vector<Point2i>& similarPixels, double similarityDistance, int imageWidth, int imageHeight)
	{
		vector<Mat> pathPoints;

		const float epsilon = 1e-8f;
		convergenceTolerance = max(convergenceTolerance, epsilon);

		Mat x = query.getMat().clone();

		pathPoints.push_back(x.clone());

		Mat_<int> indices(1, K);
		Mat_<float> distances(1, K);

		const int pixelCount = imageWidth * imageHeight;

		Mat mxo;
		int iteration = 0;
		while (iteration < maxIteration) {
			/// nearest neighbour search
			const int unlimitedSearch = -1;
			flann_index.knnSearch(x, indices, distances, K, flann::SearchParams(unlimitedSearch));
			const int count = findCount(distances, h, Kmin);
			if (count == 0) {
				break;   // converged
			}

			int* indicesArr = (int*)indices.data;
			float* distancesArr = (float*)distances.data;

			// perturb farthest neighbour point if it equals to the center point
			const float minimumRadius = 1.0f;
			const float windowRadius = sqrt(distancesArr[count - 1]);
			if (iteration == 0 && windowRadius < minimumRadius) {
				perturbPoint(x, minimumRadius);
				continue;
			}
			h = windowRadius;

			// to speed-up (find pixels with same/very similar RGB values)
			findSimilarFeatures(iteration, count, indicesArr, distancesArr, similarityDistance, imageWidth, pixelCount, similarPixels);

			/// calculate mean-shift update
			Mat x_new = meanShift(features, kernelFunctor, h, indicesArr, x, count);
			if (x_new.empty()) {
				break;	// converged
			}

			/// find mean-shift vector
			Mat mx = x_new - x;

			/// mean-shift vector convergence check (avoid very small updates, increases efficiency)
			float r = (float)norm(mx);
			if (r < convergenceTolerance) {
				break;	// converged
			}

			/// mean-shift vector length regularization  (projected to domain for constraint 1)
			applyLengthRegularization(mx, r, maximumLength);

			if (!mxo.empty()) {
				/// momentum for gradient descent
				const float gamma = 0.50f;
				applyMomentumToGradientDescent(mx, mxo, gamma);

				/// mean-shift vector direction regularization  (will be projected to domain for constraint 2)					
				const double maximumAngle = (M_PI / 4.0);
				applySmoothnessRegularization(mx, mxo, maximumAngle);
			}
			mxo = mx;

			/// move to new point
			x = x + direction * mx;

			// check the domain
			if (isOutOfDomain(x)) {
				break;
			}

			/// add to the list
			pathPoints.push_back(x.clone());

			iteration++;
		}

		return pathPoints;
	}

private:

	static int findCount(Mat_<float>& distances, float h, int minimumPointCount)
	{
		float* distancesArr = (float*)distances.data;

		double squaredRadius = (double)h * h;
		for (int i = minimumPointCount - 1; i < distances.cols; i++) {
			if (distancesArr[i] >= squaredRadius) {
				return i + 1;
			}
		}

		return minimumPointCount;
	}

	static void perturbPoint(Mat& x, float noiseLevel)
	{
		float* xArr = (float*)x.data;
		for (int i = 0; i < x.cols; i++) {
			float noise = noiseLevel * (rand() % 2 == 0 ? -1 : +1) * ((rand() % 256) / 255.0f + 1.0f);
			xArr[i] = min(max(xArr[i] + noise, 0.0f), 255.0f);
		}
	}

	static void findSimilarFeatures(int iteration, int count, int* indicesArr, float* distancesArr, double similarityDistance, int imageWidth, int pixelCount, vector<Point2i>& similarPixels)
	{
		const int firstIteration = 0;
		if (iteration == firstIteration) {
			similarPixels.clear();

			for (int i = 0; i < count; i++) {
				if (distancesArr[i] <= similarityDistance) {
					int r = indicesArr[i];
					if (r < pixelCount) {
						int xx = r % imageWidth;
						int yy = r / imageWidth;

						similarPixels.push_back(Point2i(xx, yy));
					}
				}
			}
		}
	}

	static Mat meanShift(Mat& features, KernelFunctor& kernelFunctor, double h, int* indicesArr, Mat& x, int count)
	{
		const float epsilon = 1e-8f;

		double kde = 0.0;
		Mat C_numerator(x.rows, x.cols, CV_64F, Scalar(0));
		for (int i = 0; i < count; i++) {
			int r = indicesArr[i];
			Mat x_i = features(Range(r, r + 1), Range::all());

			const double d = norm((x - x_i) / h);
			const double x_d = kernelFunctor.eval(d);

			const double sqr_x_d = sqr(x_d);

			kde += sqr_x_d;
			C_numerator += x_i * sqr_x_d;
		}
		if (kde < epsilon || norm(C_numerator) < epsilon) {
			return Mat();	// converged
		}
		Mat x_new = C_numerator / kde;
		x_new.convertTo(x_new, CV_32F);

		return x_new;
	}

	static void applyLengthRegularization(Mat& mx, float r, float maximumLength)
	{
		mx = min(r, maximumLength) * (mx / r);
	}

	static void applyMomentumToGradientDescent(Mat& mx, Mat& mxo, float gamma)
	{
		mx = gamma * mxo + (1.0f - gamma) * mx;
	}

	static void applySmoothnessRegularization(Mat& mx, Mat& mxo, double alpha)
	{
		const float epsilon = 1e-8f;

		Mat sOld3D = mxo.clone();
		Mat sNew3D = mx.clone();

		Mat sOld3DUnit = (1.0 / norm(sOld3D)) * sOld3D;
		Mat sNew3DUnit = (1.0 / norm(sNew3D)) * sNew3D;

		bool isProjectionNeeded = (sOld3DUnit.dot(sNew3DUnit) < cos(alpha));
		if (isProjectionNeeded) {
			Mat a = sOld3DUnit.cross(sNew3DUnit);
			const bool isInSameDirection = (norm(a) < epsilon);		// ???
			if (!isInSameDirection) {
				Mat R = calculateVectorAlignmentRotationMatrix(a, epsilon);

				Mat v1_hat, v2_hat;
				const bool noRotationNeeded = R.empty();
				if (noRotationNeeded) {
					// normal is already [0 0 1]
					v1_hat = sOld3D;
					v2_hat = sNew3D;
				}
				else {
					v1_hat = (R * sOld3D.t()).t();
					v2_hat = (R * sNew3D.t()).t();
				}

				// now we are working in 2D (R projects vectors into xy-plane)
				Mat sOld = v1_hat(Range(0, 1), Range(0, 2)).clone();
				Mat sNew = v2_hat(Range(0, 1), Range(0, 2)).clone();

				Mat sOldUnit = (1.0 / norm(sOld)) * sOld;
				Mat sNewUnit = (1.0 / norm(sNew)) * sNew;

				const float theta = acos(dotProduct(sOldUnit, sNewUnit));

				const float beta = (float)(alpha - theta);
				Mat sNewReg1 = (getRotationMatrix(beta) * sNew.t()).t();
				Mat sNewReg2 = (getRotationMatrix(-beta) * sNew.t()).t();

				Mat sNewReg;
				if (dotProduct(sNewReg1, sOld) > dotProduct(sNewReg2, sOld))
					sNewReg = sNewReg1;
				else
					sNewReg = sNewReg2;

				if (noRotationNeeded) {
					mx.at<float>(0, 0) = sNewReg.at<float>(0, 0);
					mx.at<float>(0, 1) = sNewReg.at<float>(0, 1);
					mx.at<float>(0, 2) = 0.0f;
				}
				else {
					Mat sNewProjected(1, 3, CV_32FC1, Scalar(0));
					sNewProjected.at<float>(0, 0) = sNewReg.at<float>(0, 0);
					sNewProjected.at<float>(0, 1) = sNewReg.at<float>(0, 1);

					mx = (R.inv() * sNewProjected.t()).t();
				}
			}
		}
	}

	// Efficiently Building a Matrix to Rotate One Vector to Another
	// https://www.tandfonline.com/doi/abs/10.1080/10867651.1999.10487509
	static Mat calculateVectorAlignmentRotationMatrix(Mat& f, double epsilon)
	{
		f = (1.0 / norm(f)) * f;
		Mat t(1, 3, CV_32FC1, Scalar(0));
		t.at<float>(0, 2) = 1.0f;

		Mat v = f.cross(t);
		double s = norm(v);

		const bool noRotationNeeded = (abs(s) < epsilon);
		if (noRotationNeeded) {
			return Mat();
		}

		Mat u = v / s;
		float c = (float)f.dot(t);
		if (abs(c) >= 1 - epsilon) {
			return Mat();
		}

		float h = (1 - c) / (1 - c * c);

		float vx = v.at<float>(0, 0);
		float vy = v.at<float>(0, 1);
		float vz = v.at<float>(0, 2);

		Mat R = Mat::eye(3, 3, CV_32FC1);

		R.at<float>(0, 0) = c + h * sqr(vx);
		R.at<float>(0, 1) = h * vx * vy - vz;
		R.at<float>(0, 2) = h * vx * vz + vy;

		R.at<float>(1, 0) = h * vx * vy + vz;
		R.at<float>(1, 1) = c + h * sqr(vy);
		R.at<float>(1, 2) = h * vy * vz - vx;

		R.at<float>(2, 0) = h * vx * vz - vy;
		R.at<float>(2, 1) = h * vy * vz + vx;
		R.at<float>(2, 2) = c + h * sqr(vz);

		return R;
	}

	static float dotProduct(Mat& a, Mat& b)
	{
		float sum = 0.0f;
		float* arr_a = (float*)a.data;
		float* arr_b = (float*)b.data;
		int n = (a.rows * a.cols);
		for (int i = 0; i < n; i++) {
			sum += arr_a[i] * arr_b[i];
		}

		return sum;
	}

	static Mat getRotationMatrix(double beta)
	{
		Mat R(2, 2, CV_32FC1);

		float cosBeta = (float)cos(beta);
		float sinBeta = (float)sin(beta);

		float* y0 = (float*)(R.data + 0 * R.step);
		float* y1 = (float*)(R.data + 1 * R.step);

		y0[0] = cosBeta;
		y0[1] = -sinBeta;
		y1[0] = sinBeta;
		y1[1] = cosBeta;

		return R;
	}

	static bool isOutOfDomain(Mat& x) {
		float* xArr = (float*)x.data;

		for (int i = 0; i < x.cols; i++) {
			if (xArr[i] < 0.0f || xArr[i] > 255.0f) {
				return true;
			}
		}

		return false;
	}

};
