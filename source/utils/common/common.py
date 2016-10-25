import os

class CommonUtils:

    def isFileExists(self,fullFilePath):
        if os.path.exists(fullFilePath):
            return True
        else:
            return False

    def getFullFilepath(self,filename):
        return os.getcwd() + filename