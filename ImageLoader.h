#pragma once
#include "OSDefines.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


class ImageLoader {
public:
	virtual Mat load(path filePath) = 0;

	vector<path> getUnreadImages()
	{
		return unreadImages;
	}

	void addImageLoadError(path filePath) {
		#pragma omp critical
		{
			unreadImages.push_back(filePath);
		}
	}

	void saveUnreadedImages(string logFileName, bool showInCosole = true)
	{
		if (unreadImages.size() > 0) {
			std::ofstream logFile(logFileName);

			if (showInCosole) {
				cout << endl << endl << endl;
				cout << "OpenCV could not load below images" << endl;
				cout << "----------------------------------" << endl;
			}

			for (int i = 0; i < unreadImages.size(); i++) {
				path filePath = unreadImages.at(i);
				logFile << filePath.relative_path() << endl;

				if (showInCosole) {
					cout << filePath.relative_path() << endl;
				}
			}

			logFile.close();
		}
		else
			remove(logFileName);
	}

protected:
	vector<path> unreadImages;

	string getFileName(path imagePath)
	{
		string fileDirectory = imagePath.parent_path().string();
		string imageFileName = imagePath.stem().string();
		string extension = imagePath.extension().string();

		string inputFileName = fileDirectory + SLASH + imageFileName + extension;

		return inputFileName;
	}

private:

};
