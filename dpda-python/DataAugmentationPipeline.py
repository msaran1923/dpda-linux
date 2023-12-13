class DataAugmentationPipeline:
    def __init__(self):
        self.dataAugmenters = []

    def execute(self, input_directory, output_directory, image_path, augmentation_count, scale_factor, percentage):
        augmentationPipeline = DataAugmentationPipeline()
        # augmentationPipeline.add(DataAugmentationDistributionPreserving(), 100)
        # augmentationPipeline.add(DataAugmenterSmoothRandomErase(), 50)
        # augmentationPipeline.add(DataAugmenterRandomErase(), 100)
