#define _USE_MATH_DEFINES

#include <chrono>
#include "FileRepositoryDirectoryBased.h"
#include "DataAugmenterDistributionPreserving.h"
#include "DataAugmenterRandomErase.h"
#include "DataAugmenterGammaCorrection.h"
#include "DataAugmenterHistogramEqualization.h"
#include "DataAugmenterFlip.h"
#include "NoiseGeneratorPerlin.h"
#include "ImageLoaderOpenCV.h"
#include "ImageSaverOpenCV.h"

using namespace std;


int main(int argc, char* argv[])
{
    cout << "Distribution-preserving Data Augmentation (DPDA) v1.05h" << endl << endl;

    vector<string> directoryNames;
    directoryNames.push_back("train");
    //directoryNames.push_back("val");  // add directories as you wish
    //directoryNames.push_back("test");


    const string inputDirectory = "images";
    const string outputDirectory = "results";
    FileRepositoryDirectoryBased fileRepository(inputDirectory, outputDirectory, directoryNames);
    vector<path> imagePaths = fileRepository.getImagePaths();


    /// augment images - begin
    auto t1 = std::chrono::high_resolution_clock::now();

    ImageLoaderOpenCV imageLoader;
    ImageSaverOpenCV imageSaver;
    NoiseGeneratorPerlin noiseGenerator;
    const float DPDA_Power = 1.0f;   // set DPDA power - 0.0f to 1.0f
    DataAugmenterDistributionPreserving dataAugmenterDistributionPreserving(&imageLoader, &imageSaver , &noiseGenerator, DPDA_Power);
    DataAugmenterRandomErase dataAugmenterRandomErase(&imageLoader, &imageSaver);
    DataAugmenterHistogramEqualization dataAugmenterHistogramEqualization(&imageLoader, &imageSaver);
    DataAugmenterGammaCorrection dataAugmenterGammaCorrection(&imageLoader, &imageSaver);
    DataAugmenterFlip dataAugmenterFlip(&imageLoader, &imageSaver);

    //Mix augmentation methods by addind to pipeline
    // InputImage -->  DataAugmenterFlip --> DataAugmenterGammaCorrection  -->  DataAugmenterRandomErase --> AugmentedImage
    //First, create a dataAugmenter with the first augmentation method you wish
    //Then, add another augmentation methos to the pipeline
    //e.g. To combine Flip, GamaCorrectin, and Random erase you should write:
    //DataAugmenter& dataAugmenter = dataAugmenterFlip;
    //dataAugmenter.setPipelineDataAugmenter(&dataAugmenterGammaCorrection);
    //dataAugmenterGammaCorrection.setPipelineDataAugmenter(&dataAugmenterRandomErase);

    //To augment images only with Distribution Preserving Data Augmentation (DPDA), use this statement
    DataAugmenter& dataAugmenter = dataAugmenterDistributionPreserving;
    //Uncomment the pipeline statements if you want to mix up augmentation methods
    //dataAugmenter.setPipelineDataAugmenter(&dataAugmenterFlip);
    //dataAugmenter.setPipelineDataAugmenter(&dataAugmenterRandomErase);
    //dataAugmenter.setPipelineDataAugmenter(&dataAugmenterGammaCorrection);




    //#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < imagePaths.size(); i++) {
        path filePath = imagePaths.at(i);

        const int augmentationCount = 5;
        const double scaleFactor = 1.0;
        const int augmentationPercentage = 100;

        dataAugmenter.execute(inputDirectory, outputDirectory, imagePaths.at(i), augmentationCount, scaleFactor, augmentationPercentage);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;
    cout << endl << endl << "Elapsed time: " << duration << " seconds \n\n";
    /// augment images - end

    string logFileName = "_imageWithProblems.txt";
    imageLoader.saveUnreadedImages(logFileName);

    return 0;
}
