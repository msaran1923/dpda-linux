#pragma once
#include "OSDefines.h"
#include <opencv2/opencv.hpp>
#include "PerlinNoise.h"
#include "NoiseGenerator.h"

using namespace cv;


class NoiseGeneratorPerlin : public NoiseGenerator {
public:
	Mat create(int width, int height, double roughness)
	{
		PerlinNoise perlinNoiseCreator(getSeed());

		Mat perlinNoise(height, width, CV_8UC1);

		double noiseScale = randRange(0.1, 10.0);
		double noiseCenter = randRange(0.35, 0.65);

		for (int y = 0; y < perlinNoise.rows; y++) {
			unsigned char* irow = (unsigned char*)(perlinNoise.data + y * perlinNoise.step);

			for (int x = 0; x < perlinNoise.cols; x++) {
				double xf = (double)x / width;
				double yf = (double)y / height;

				double noise = perlinNoiseCreator.noise(xf * roughness, yf * roughness, 1.0);
				double adjustedNoise = (tanh(noiseScale * (noise - noiseCenter)) + 1.0) / 2.0;

				irow[x] = (unsigned char)(255 * adjustedNoise);
			}
		}

		return perlinNoise;
	}

private:
	int width;
	int height;
	double roughness;

	int getSeed() {
		static unsigned int runningIndex = 0;
		runningIndex++;

		return ((unsigned)time(0) + runningIndex);
	}

	inline double randRange(double minValue, double maxValue)
	{
		const int randomRange = 8192;
		return (rand() % randomRange) / (randomRange - 1.0) * (maxValue - minValue) + minValue;
	}

};


