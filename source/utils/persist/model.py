import pandas
from sklearn import tree, svm
from sklearn.externals import joblib
import numpy as np

class Model:

    def saveModel(self, trainingDataPath):
        dataframeTraining = pandas.read_csv(trainingDataPath)
        columnsTraining = len(dataframeTraining.columns)
        columnIndex = columnsTraining - 1
        arrayTraining = dataframeTraining.values
        X = arrayTraining[:, 0:columnIndex]
        Y = arrayTraining[:, columnIndex]
        test_size = 0.33
        seed = columnIndex - 1

        # Fit the model on 33%
        decisionTreeModel = tree.DecisionTreeClassifier()
        decisionTreeModel.fit(X, Y)
        # save the model to disk
        filename = 'DecisionTree.sav'
        joblib.dump(decisionTreeModel, filename)

        svmModel = svm.SVC()
        svmModel.fit(X,Y)
        joblib.dump(svmModel, 'SVMTree.sav')

    def tuneModel(self, validationDataPath):
        dataframeValidation = pandas.read_csv(validationDataPath)
        columnsTraining = len(dataframeValidation.columns)
        columnIndex = columnsTraining - 1
        arrayTraining = dataframeValidation.values
        X = arrayTraining[:, 0:columnIndex]
        Y = arrayTraining[:, columnIndex]
        print("Actual Ans: ")
        print(Y)
        treeModel = joblib.load('DecisionTree.sav')
        treePrediction = np.array(treeModel.predict(X))
        print("Tree Prediction: ")
        print(treePrediction)
        treeScore = treeModel.score(X, Y)


        svmModel = joblib.load('SVMTree.sav')
        svmPrediction = np.array(svmModel.predict(X))
        print("SVM Prediction")
        print(svmPrediction)
        svmScore = svmModel.score(X, Y)
        print("Accuracy: %0.2f " % treeScore)
        print("Accuracy: %0.2f " % svmScore)

        return 'DecisionTree.sav' if (treeScore > svmScore) else 'SVMTree.sav'

    def loadModel(self, validationDataPath, inputDataPath):
        preferredModel = self.tuneModel(validationDataPath)
        dataframeML = pandas.read_csv(inputDataPath)
        ml_data_array = dataframeML.values
        loaded_model = joblib.load(preferredModel)
        result = np.array([loaded_model.predict(ml_data_array)])
        result = np.reshape(result, (len(ml_data_array),1))
        output = np.append(ml_data_array, result, axis=1)
        np.savetxt('iris_output.csv', output, fmt='%.1f', delimiter=', ')