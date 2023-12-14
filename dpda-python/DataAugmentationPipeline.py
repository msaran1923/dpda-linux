from DataAugmenterDistributionPreserving import DataAugmenterDistributionPreserving
from ImageLoaderOpenCV import ImageLoaderOpenCV
from ImageSaverOpenCV import ImageSaverOpenCV
from NoiseGeneratorPerlin import NoiseGeneratorPerlin

class DataAugmentationPipeline:
    def __init__(self):
        imageLoader = ImageLoaderOpenCV()
        imageSaver = ImageSaverOpenCV()
        noiseGenerator = NoiseGeneratorPerlin()
        DPDA_Power = 1.0
        dataAugmenterDistributionPreserving = DataAugmenterDistributionPreserving(imageLoader, imageSaver, noiseGenerator, DPDA_Power)
        self.dataAugmenters = [dataAugmenterDistributionPreserving]

    def appendToPipeline(self, nextAugmenter):
        self.dataAugmenters.append(nextAugmenter)
        if len(self.dataAugmenters) > 1:
            self.dataAugmenters[-2].pipelineDataAugmenter = self.dataAugmenters[-1]

    def execute(self, inputDirectory, outputDirectory, imagePath, augmentationCount, scaleFactor, percentage):
        self.dataAugmenters[0].execute(inputDirectory, outputDirectory, imagePath, augmentationCount, scaleFactor, percentage)
