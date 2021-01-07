# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 03:55:52 2021

@author: Hossein
"""
from MachineLearning import MyMachine
from MachineLearning import fileIO
from Recognizer import Recognizer

Train = False
Perdict = ~Train

def Main():
    if(Train):
        print("Loading CSV...")
        machine = MyMachine("./Data/labels.csv")
        print("CSV Loaded.")
        trainCoulumnName = "id"
        valuesColumnName = "breed"
        filesType = ".jpg"
        TrainingPicturesPathDirectory = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\Data\\train\\"
        TestingPicturesPathDirectoy= "./Data/test/"
        trainingImagesNo = 12000 # Should be more than 126 due to __show_25_images() u may get Index out of bound error
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
    else:
        print("Recognizing...")
        imgPath = "6.jpg"
        # Predict an Image
        modelPath = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\Models\\ModelNO_10_LOSS_0.74_ACCURACY_0.82_IMG_NO_12000.h5" 
        CSVLabelsPath = "./Data/labels.csv"
        X_ColumnName = "id"
        Y_ColumnName = "breed"
        imagePath = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\ImagesToTest\\" + str(imgPath)
        trainPicturesPath = "C:\\Users\\hosse\\Desktop\\Dog Breed Project\\Data\\train\\"
        filesType = ".jpg"
        imgReco = Recognizer(modelPath,CSVLabelsPath,X_ColumnName,Y_ColumnName,trainPicturesPath,filesType)
        imgReco.predict(imagePath)
        
        
    return 0

Main()
#fileIO("Counter.txt")