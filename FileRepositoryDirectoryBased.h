#pragma once
#include "OSDefines.h"
#include <iostream>
#include "FileRepository.h"
#include "OSDefines.h"

using namespace std;


class FileRepositoryDirectoryBased : public FileRepository {
public:
	FileRepositoryDirectoryBased(string inputDirectory, string outputDirectory, vector<string>& directoryNames) 
	{
		buildPaths(inputDirectory, outputDirectory, directoryNames);
	}

	vector<path> getImagePaths() 
	{
		return imagePaths;
	}

private:
	vector<path> imagePaths;

	void buildPaths(string inputDirectory, string outputDirectory, vector<string>& directoryNames)
	{
		for (int i = 0; i < directoryNames.size(); i++) {
			string resultDirectoryName = outputDirectory + SLASH + directoryNames.at(i);
			create_directory(resultDirectoryName);

			string directoryName = inputDirectory + SLASH + directoryNames.at(i);
			for (directory_iterator itr(directoryName); itr != directory_iterator(); ++itr) {
				path classPath = itr->path();
				if (is_directory(classPath)) {
					create_directory(resultDirectoryName + SLASH + classPath.filename().string());

					string imageDirectoryName = classPath.parent_path().string() + SLASH + classPath.filename().string();

					for (directory_iterator itrImages(imageDirectoryName); itrImages != directory_iterator(); ++itrImages) {
						path imagePath = itrImages->path();

						if (isImage(imagePath.extension().string())) {
							imagePaths.push_back(imagePath);
						}
					}
				}
			}
		}
	}

};
