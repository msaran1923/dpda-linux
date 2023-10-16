import os
from FileRepository import FileRepository

class FileRepositoryDirectoryBased(FileRepository):
    def __init__(self, inputDirectory, outputDirectory, directoryNames):
        self.imagePaths = []
        self.buildPaths(inputDirectory, outputDirectory, directoryNames)

    def getImagePaths(self):
        return self.imagePaths

    def buildPaths(self, inputDirectory, outputDirectory, directoryNames):
        for directoryName in directoryNames:
            resultDirectoryName = os.path.join(outputDirectory, directoryName)
            os.makedirs(resultDirectoryName, exist_ok=True)

            directoryPath = os.path.join(inputDirectory, directoryName)
            for root, dirs, files in os.walk(directoryPath):
                for filename in files:
                    imagePath = os.path.join(root, filename)
                    if self.isImage(os.path.splitext(imagePath)[1].lower()):
                        self.imagePaths.append(imagePath)
