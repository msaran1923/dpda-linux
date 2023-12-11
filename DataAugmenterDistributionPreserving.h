#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include <opencv2/hdf.hpp>
#include "FeatureExtractor.h"
#include "DensityDecreasingPath.h"
#include "DataAugmenter.h"
#include "NoiseGenerator.h"

using namespace std;
using namespace cv;


class DataAugmenterDistributionPreserving : public DataAugmenter {
public:
	DataAugmenterDistributionPreserving(ImageLoader* imageLoader, ImageSaver* imageSaver, NoiseGenerator* noiseGenerator, float DPDA_Power)
		: DataAugmenter(imageLoader, imageSaver)
	{
		this->noiseGenerator = noiseGenerator;
		this->DPDA_Power = DPDA_Power;
	}

	Mat findCoefficients(vector<Mat>& x, int dimensionIndex)
	{
		const int n = (int)x.size();

		Mat A(n, n, CV_64F, Scalar(1.0));
		for (int i = 0; i < n; i++) {
			double* A_i = (double*)(A.data + i * A.step);
			A_i[i] = 4.0;
		}

		Mat f(n, 1, CV_64F, Scalar(0.0));
		double* f_p = (double*)f.data;
		for (int i = 1; i < n - 1; i++) {
			double y_ip = *((float*)(x.at(i - 1).data) + dimensionIndex);
			double y_i = *((float*)(x.at(i).data) + dimensionIndex);
			double y_in = *((float*)(x.at(i + 1).data) + dimensionIndex);

			f_p[i] = 6.0 * ((y_in - y_i) - (y_i - y_ip));
		}

		Mat s = A.inv() * f;
		double* s_p = (double*)s.data;

		const int degree = 3;
		Mat abcd(n - 1, degree + 1, CV_64F);
		for (int i = 0; i < n - 1; i++) {
			double* abcd_arr = (double*)(abcd.data + i * abcd.step);

			double y_i = *((float*)(x.at(i).data) + dimensionIndex);
			double y_in = *((float*)(x.at(i + 1).data) + dimensionIndex);

			abcd_arr[0] = (s_p[i + 1] - s_p[i]) / 6.0;
			abcd_arr[1] = s_p[i] / 2.0;
			abcd_arr[2] = (y_in - y_i) - (2 * s_p[i] + s_p[i + 1]) / 6.0;
			abcd_arr[3] = y_i;
		}

		return abcd;
	}

	vector<Mat> regularizePoints(vector<Mat>& points, int labelCount)
	{
		vector<Mat> refinedPoints;

		const int n = (int)points.size();

		// repeats the point labelCount times
		if (n == 1) {
			for (int i = 0; i < labelCount; i++) {
				refinedPoints.push_back(points.at(0).clone());
			}
		}

		// linearly interpolates between first and last point
		if (n == 2) {
			Mat pts1 = points.at(0);
			Mat pts2 = points.at(1);
			for (int i = 0; i < labelCount; i++) {
				const double alpha = i / (labelCount - 1.0);
				Mat interpolatedPoint = (1.0f - alpha) * pts1 + alpha * pts2;
				refinedPoints.push_back(interpolatedPoint);
			}
		}

		// cubic spline interpolation between points
		if (n >= 3) {
			Mat abcd_x = findCoefficients(points, 0);
			Mat abcd_y = findCoefficients(points, 1);
			Mat abcd_z = findCoefficients(points, 2);

			for (int i = 0; i < labelCount; i++) {
				double t = i / (labelCount - 1.0) * (n - 1.0);
				int t_i = (int)t;
				double i_d = t - t_i;

				if (t_i == n - 1 && i_d == 0) {
					refinedPoints.push_back(points.at(n - 1));
					break;
				}

				double* abcd_x_i = (double*)(abcd_x.data + t_i * abcd_x.step);
				double& ax_i = abcd_x_i[0];
				double& bx_i = abcd_x_i[1];
				double& cx_i = abcd_x_i[2];
				double& dx_i = abcd_x_i[3];

				double* abcd_y_i = (double*)(abcd_y.data + t_i * abcd_y.step);
				double& ay_i = abcd_y_i[0];
				double& by_i = abcd_y_i[1];
				double& cy_i = abcd_y_i[2];
				double& dy_i = abcd_y_i[3];

				double* abcd_z_i = (double*)(abcd_z.data + t_i * abcd_z.step);
				double& az_i = abcd_z_i[0];
				double& bz_i = abcd_z_i[1];
				double& cz_i = abcd_z_i[2];
				double& dz_i = abcd_z_i[3];

				double x_c = ax_i * cube(i_d) + bx_i * sqr(i_d) + cx_i * i_d + dx_i;
				double y_c = ay_i * cube(i_d) + by_i * sqr(i_d) + cy_i * i_d + dy_i;
				double z_c = az_i * cube(i_d) + bz_i * sqr(i_d) + cz_i * i_d + dz_i;

				Mat interpolatedPoint(1, 3, CV_32F);
				float* interpolatedPoint_p = (float*)(interpolatedPoint.data);
				interpolatedPoint_p[0] = (float)x_c;
				interpolatedPoint_p[1] = (float)y_c;
				interpolatedPoint_p[2] = (float)z_c;
				refinedPoints.push_back(interpolatedPoint);
			}
		}

		return refinedPoints;
	}

	bool isEmpty(RGB<float>* allPathPointsData, int dataCount) {
		for (int i = 0; i < dataCount; i++) {
			if (allPathPointsData[i].red > 0 && allPathPointsData[i].green > 0 && allPathPointsData[i].blue > 0) {
				return false;
			}
		}

		return true;
	}

	Mat createDensityDecreasingPath(Mat& image, float hInitial, int L, Mat& features, flann::Index& flann_index, KernelFunctor& kernelFunctor,
		int d, int K, float convergenceTolerance, float maximumLength)
	{
		Mat allPathPoints(image.rows * image.cols, L, CV_32FC3, Scalar(0));

		Mat_<float> query(1, d);
		float* queryArr = (float*)query.data;

		const float direction = -1.0f;
		float h = hInitial;

		int noPathPointCount = 0;

		int pixelIndex = 0;
		for (int y = 0; y < image.rows; y++) {
			RGB<uchar>* irow = (RGB<uchar>*)(image.data + y * image.step);

			for (int x = 0; x < image.cols; x++) {
				RGB<float>* allPathPointsData = (RGB<float>*)(allPathPoints.data + pixelIndex * allPathPoints.step);

				// if already found then skip
				if (isEmpty(allPathPointsData, L)) {
					queryArr[0] = irow[x].red;
					queryArr[1] = irow[x].green;
					queryArr[2] = irow[x].blue;

					const int minimumPointCount = 32;
					double similarityDistance = d * sqr(2.0);
					vector<Point2i> similarPixels;
					vector<Mat> unregularPathPoints = DensityDecreasingPath::findPath(features, flann_index, K, minimumPointCount, h, query,
						convergenceTolerance, L/2, direction, maximumLength, kernelFunctor,
						similarPixels, similarityDistance, image.cols, image.rows);

					auto pathPointCount = unregularPathPoints.size();
					if (pathPointCount <= 1) {
						//ut << "P(" << x << ", " << y << ") = " << (int)irow[x].red << " " << (int)irow[x].green << " " << (int)irow[x].blue << endl;
						noPathPointCount++;
					}

					vector<Mat> pathPoints = regularizePoints(unregularPathPoints, L);

					for (int i = 0; i < pathPoints.size(); i++) {
						RGB<float>* sourcePtr = (RGB<float>*)pathPoints.at(i).data;
						allPathPointsData[i] = *sourcePtr;		// assigns red, blue and green via operator=
					}

					// for similarPixels use obtained density-decreasing centers to speed-up
					for (int k = 0; k < similarPixels.size(); k++) {
						Point2i pts = similarPixels.at(k);
						const int pixelIndex = pts.y * image.cols + pts.x;
						RGB<float>* similarPathPointsData = (RGB<float>*)(allPathPoints.data + pixelIndex * allPathPoints.step);

						for (int i = 0; i < pathPoints.size(); i++) {
							RGB<float>* sourcePtr = (RGB<float>*)pathPoints.at(i).data;
							similarPathPointsData[i] = *sourcePtr;		// assigns red, blue and green via operator=
						}
					}

					h = max(h * 0.99f, hInitial);
				}

				pixelIndex++;
			}
			cout << " [" << h << "] ";
		}
		cout << endl << 100.0 * (double)noPathPointCount / (image.cols * image.rows) << "% no density-decrease" << endl << endl;

		return allPathPoints;
	}

	vector<Mat> createAugmentedImages(Mat& image, Mat& allPathPoints, int d, int labelCount, int augmentationCount, vector<bool> applyDPDA_Decisions)
	{
		vector<Mat> augmentedImages;

		Mat_<float> query(1, d);
		float* queryArr = (float*)query.data;

		for (int i = 0; i < augmentationCount; i++) {
			bool applyDPDA = !allPathPoints.empty() && applyDPDA_Decisions.at(i);

			const float DPDA_Baseline = (float)randUnity();
			const float brightnessBaseline = 1.0f - DPDA_Baseline;
			//cout << "DPDA_Baseline = " << DPDA_Baseline << ", brightnessBaseline = " << brightnessBaseline << endl;

			double perlinRoghness = randRange(1.0, 5.0);
			Mat perlinNoise = noiseGenerator->create(image.cols, image.rows, perlinRoghness);

			Mat augmentedImage(image.rows, image.cols, CV_8UC3, Scalar(0));

			int pixelIndex = 0;
			for (int y = 0; y < image.rows; y++) {
				float* queryArr = (float*)query.data;

				RGB<uchar>* irow = (RGB<uchar>*)(image.data + y * image.step);

				unsigned char* pmrow = (unsigned char*)(perlinNoise.data + y * perlinNoise.step);
				RGB<uchar>* airow = (RGB<uchar>*)(augmentedImage.data + y * augmentedImage.step);

				for (int x = 0; x < image.cols; x++) {
					RGB<float> rgb;
					rgb.red = irow[x].red;
					rgb.green = irow[x].green;
					rgb.blue = irow[x].blue;

					if (applyDPDA) {
						const float DPDA_Effect = 1.0f - (pmrow[x] / 255.0f);	// Perlin noise, per-pixel
						const float noiseIndexDPDA = min(DPDA_Baseline + DPDA_Effect, 1.0f);
						const int DPDA_Index = min((int)((labelCount - 1.0) * noiseIndexDPDA), labelCount - 1);
						RGB<float>* DPDA_Data = (RGB<float>*)(allPathPoints.data + pixelIndex * allPathPoints.step);
						rgb = DPDA_Data[DPDA_Index];		// assigns red, blue and green via operator=
					}

					// assign result
					airow[x] = rgb;

					pixelIndex++;
				}
			}

			augmentedImages.push_back(augmentedImage);
		}

		return augmentedImages;
	}

	vector<Mat> distributionPreservingDataAugmentation(Mat& image, int augmentationCount, 
		float DPDA_Power, int augmentationPercentage, const String imageFileName) {

		// create color features
		int d = image.channels();
		Mat features = FeatureExtractor::create(image);

		// kd-tree search test
		const int K = 256;
		const int L = 64;
		const float convergenceTolerance = 0.01f * d;
		const float hInitial = estimateH(image);
		const float maximumLength = DPDA_Power * sqrt(hInitial);

		// constructs flann tree (approximate kd-tree search)
		int treeCount = 1;	// min(max(omp_get_max_threads(), 1), 32);
		flann::Index flann_index(features, flann::KDTreeIndexParams(treeCount), cvflann::FLANN_DIST_EUCLIDEAN);

		vector<bool> applyDPDA_Decisions;
		bool atLeastOneDPDA_Application = false;
		for (int i = 0; i < augmentationCount; i++) {
			bool applyDPDA = (rand() % 100 < augmentationPercentage);
			applyDPDA_Decisions.push_back(applyDPDA);
			if (applyDPDA) {
				atLeastOneDPDA_Application = true;
			}
		}

		EpanechnikovKernel kernelFunctor;
		Mat allPathPoints;
		if (atLeastOneDPDA_Application) {
			if (boost::filesystem::exists("pathpoints/pathpoint_" + imageFileName + ".hdf5"))
				allPathPoints = allPathPointsLoader(imageFileName);
			else {
				allPathPoints = createDensityDecreasingPath(image, hInitial, L, features, flann_index, kernelFunctor,
													 	d, K, convergenceTolerance, maximumLength);
				allPathPointsWriter(allPathPoints, imageFileName);
			}
		}

		vector<Mat> augmentedImages = createAugmentedImages(image, allPathPoints, d, L, augmentationCount, applyDPDA_Decisions);

		return augmentedImages;
	}

	void allPathPointsWriter(const Mat& allPathPoints, String imageFileName) {
		String fileName = "pathpoints/pathpoint_" + imageFileName + ".hdf5";
		String datasetName = "allPathPoints";
		
		int sizes[] = {allPathPoints.rows, allPathPoints.cols, 3};
		Mat apptmp(3, sizes, CV_32FC1);
		
		for (int i = 0; i < allPathPoints.rows; ++i) {
			for (int j = 0; j < allPathPoints.cols; ++j) {
				RGB<float> elem = allPathPoints.at<RGB<float>>(i, j);
				apptmp.at<cv::Vec3f>(i, j) = cv::Vec3f(elem.red, elem.green, elem.blue);
			}
		}

		Ptr<hdf::HDF5> h5io = hdf::open(fileName);
		h5io->dswrite(apptmp, datasetName);
		h5io->close();
	}

	Mat allPathPointsLoader(String imageFileName) {
		String fileName = "pathpoints/pathpoint_" + imageFileName + ".hdf5";
		String datasetName = "allPathPoints";
		
		Mat apptmp;
		cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(fileName);
		h5io->dsread(apptmp, datasetName);
		h5io->close();

		int sizes[] = {apptmp.size[0], apptmp.size[1]};
		Mat allPathPoints(2, sizes, CV_32FC3, Scalar(0));
		
		for (int i = 0; i < allPathPoints.rows; ++i) {
			for (int j = 0; j < allPathPoints.cols; ++j) {
				allPathPoints.at<RGB<float>>(i, j) = apptmp.at<RGB<float>>(i, j);
			}
		}

		return allPathPoints;
	}

	bool execute(String inputDirectory, String outputDirectory, path& imagePath,
		int augmentationCount, double scaleFactor, int augmentationPercentage)
	{
		String fileDirectory = imagePath.parent_path().string();
		String imageFileName = imagePath.stem().string();
		String extension = imagePath.extension().string();

		String inputFileName = fileDirectory + SLASH + imageFileName + extension;

		Mat image = imageLoader->load(inputFileName);
		if (image.empty()) {
			cout << endl;
			return false;
		}

		Mat resizedImage;
		resize(image, resizedImage, Size(), scaleFactor, scaleFactor);

		// process (create augmented images)
		auto t1 = std::chrono::high_resolution_clock::now();

		// DPDA augmentation
		vector<Mat> augmentedImages = distributionPreservingDataAugmentation(resizedImage, augmentationCount, DPDA_Power, augmentationPercentage, imageFileName);

		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;

		#pragma omp critical
		{
			cout << ",   Elapsed time: " << duration << " seconds \n";
		}

		// write original image
		String outputFileName = fileDirectory + SLASH + imageFileName + extension;
		boost::replace_all(outputFileName, inputDirectory, outputDirectory);
		imageSaver->save(resizedImage, outputFileName);

		// write augmented images (x count)
		for (int k = 0; k < augmentedImages.size(); k++) {
			Mat augmentedImage = augmentedImages.at(k);

			stringstream ss;
			ss << fileDirectory << SLASH << imageFileName << "_" << k << extension;
			String outputFileName = ss.str();
			boost::replace_all(outputFileName, inputDirectory, outputDirectory);

			augmentedImage = augmentImage(augmentedImage);
			imageSaver->save(augmentedImage, outputFileName);
		}

		return true;
	}

	Mat augmentImage(Mat image) {
		if (pipelineDataAugmenter == 0)
			return image;
		else
			return pipelineDataAugmenter->augmentImage(image);
	}

private:
	NoiseGenerator* noiseGenerator;
	float DPDA_Power;

	float estimateH(Mat& image)
	{
		vector<int> distances;

		const int maxD = 256 * 256;

		for (int y = 1; y < image.rows - 1; y++) {
			RGB<uchar>* irow = (RGB<uchar>*)(image.data + y * image.step);
			RGB<uchar>* urow = (RGB<uchar>*)(image.data + (y - 1) * image.step);
			RGB<uchar>* drow = (RGB<uchar>*)(image.data + (y + 1) * image.step);

			for (int x = 1; x < image.cols - 1; x++) {
				RGB<uchar> c = irow[x];
				RGB<uchar> c1 = irow[x - 1];
				RGB<uchar> c2 = irow[x + 1];
				RGB<uchar> c3 = urow[x];
				RGB<uchar> c4 = drow[x];

				int d1 = sqr((int)c.red - c1.red) + sqr((int)c.green - c1.green) + sqr((int)c.blue - c1.blue);
				int d2 = sqr((int)c.red - c2.red) + sqr((int)c.green - c2.green) + sqr((int)c.blue - c2.blue);
				int d3 = sqr((int)c.red - c3.red) + sqr((int)c.green - c3.green) + sqr((int)c.blue - c3.blue);
				int d4 = sqr((int)c.red - c4.red) + sqr((int)c.green - c4.green) + sqr((int)c.blue - c4.blue);
				const int d = min(min(d1, d2), min(d3, d4));

				if (d > 1 && d < maxD) {
					distances.push_back(d);
				}
			}
		}

		const auto median_it = distances.begin() + distances.size() / 2;
		std::nth_element(distances.begin(), median_it, distances.end());
		auto median = *median_it;

		return max((float)sqrt(median), 1.0f);
	}

};
