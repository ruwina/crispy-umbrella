import sys

from source.utils.common import common
from source.utils.persist import model

def main():
    training = sys.argv[1] if len(sys.argv) > 1 else '/data/iris_training.csv'
    validation = sys.argv[2] if len(sys.argv) > 2 else '/data/iris_validation.csv'
    input = sys.argv[3] if len(sys.argv) > 3 else '/data/iris_input.csv'

    commonObj = common.CommonUtils()
    trainingPath = commonObj.getFullFilepath(training)
    validationPath = commonObj.getFullFilepath(validation)
    inputPath = commonObj.getFullFilepath(input)

    treeModel = commonObj.getFullFilepath('DecisionTree.sav')
    svmModel = commonObj.getFullFilepath('SVMTree.sav')

    modelObj = model.Model()

    if((commonObj.isFileExists(treeModel) == False) and (commonObj.isFileExists(svmModel) == False)):
        modelObj.saveModel(trainingPath)

    modelObj.loadModel(validationPath, inputPath)

if __name__ == "__main__":
    main()