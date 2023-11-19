class FileRepository():
    def getImagePaths(self):
        pass

    def isImage(self, extension):
        return extension in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
