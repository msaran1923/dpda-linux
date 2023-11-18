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
            
            directoryName = os.path.join(inputDirectory, directoryName)
            for path in os.listdir(directoryName):
                classPath = os.path.join(directoryName, path)
                if os.path.isdir(classPath):
                    os.makedirs(os.path.join(resultDirectoryName, path), exist_ok=True)
                    for imagePath in os.listdir(classPath):
                        if self.isImage(os.path.splitext(imagePath)[1]):
                            self.imagePaths.append(os.path.join(classPath, imagePath))
