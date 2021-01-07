# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 03:55:52 2021

@author: Hossein
"""
from MachineLearning import MyMachine
from MachineLearning import fileIO

def Main():
    print("Loading CSV...")
    machine = MyMachine("./Data/labels.csv")
    print("CSV Loaded.")
    trainCoulumnName = "id"
    valuesColumnName = "breed"
    filesType = ".jpg"
    TrainingPicturesPathDirectory = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\Data\\train\\"
    TestingPicturesPathDirectoy= "./Data/test/"
    trainingImagesNo = 126 # Should be more than 126 due to __show_25_images() u may get Index out of bound error
    callbackLogsPath = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\Logs\\"
    modelsSavingPath = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\Models\\"
    


    machine.initializer(trainCoulumnName,
                        valuesColumnName,
                        filesType,
                        TrainingPicturesPathDirectory,
                        TestingPicturesPathDirectoy,
                        trainingImagesNo,
                        callbackLogsPath,
                        modelsSavingPath)
    
    machine.Train()

    return 0

Main()
#fileIO("Counter.txt")