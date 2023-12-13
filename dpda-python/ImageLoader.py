import cv2
import os

class ImageLoader:
    def __init__(self):
        self.unreadImages = []

    def load(self, filePath):
        try:
            image = cv2.imread(str(filePath))
            if image is None:
                self.addImageLoadError(filePath)
            return image
        except Exception as e:
            print(f'Error loading image {filePath}: {str(e)}')
            self.addImageLoadError(filePath)
            return None

    def getUnreadImages(self):
        return self.unreadImages

    def addImageLoadError(self, filePath):
        self.unreadImages.append(filePath)

    def saveUnreadedImages(self, logFileName, showInConsole=True):
        if self.unreadImages:
            with open(logFileName, 'w') as logFile:
                if showInConsole:
                    print('\nOpenCV could not load below images')
                    print('----------------------------------')

                for filePath in self.unreadImages:
                    logFile.write(str(filePath) + "\n")

                    if showInConsole:
                        print(filePath)

        else:
            if os.path.exists(logFileName):
                os.remove(logFileName)

    def getFileName(self, imagePath):
        fileDirectory, fn = os.path.split(imagePath)
        imageFileName, extension = os.path.splitext(fn)

        inputFileName = fileDirectory + '/' + imageFileName + extension
        return inputFileName
