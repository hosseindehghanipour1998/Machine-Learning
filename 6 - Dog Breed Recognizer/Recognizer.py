# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 18:35:10 2021

@author: Hossein
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
from IPython.display import Image
import os
import matplotlib.pyplot as plt
import datetime



class Recognizer:
    def __init__(self, modelPath, CSVLabelsPath, X_ColumnName, Y_ColumnName, trainPicturesPath, filesType):
        
        # ---------------- FETCH LABELS -----------------
        self.labels_csv = pd.read_csv(CSVLabelsPath)
        self.labels = self.labels_csv[Y_ColumnName].to_numpy() 
        self.unique_values = np.unique(self.labels)
        
        # ------------------ INITIALIZER -----------------
        self.__modelPath = modelPath
        self.__IMAGE_SIZE = 224
        self.__BATCH_SIZE = 32
        self.__Y_ColumnName = Y_ColumnName
        self.__X_ColumnName = X_ColumnName
        self.__picturesPathDirectory = trainPicturesPath
        self.__files_type = filesType
        # ----------------- METHODS ----------------------
        self.filesNames = None
        self.__model = None
    
    def __load_model(self,model_path):
      """
      Loads a saved model from a specified path.
      """
      print(f"Loading saved model from: {model_path}")
      model = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer":hub.KerasLayer})
      return model
  
    def __predict_a_picture(self,imagePath, imgSize, ourModel, batch_size):
      #image = preporcessImage(imagePath,imgSize)

      data = tf.data.Dataset.from_tensor_slices(tf.constant([imagePath]))
      data_batch = data.map(self.__preporcessImage).batch(batch_size)
      #print(data_batch)
      my_prediction = ourModel.predict(data_batch)
    
      IMAGE_INDEX = 0 
      max_prediction = np.max(my_prediction[IMAGE_INDEX])
      index_with_max_prediction = np.argmax(my_prediction[IMAGE_INDEX])
      
      predicted_dog_breed = self.unique_values[index_with_max_prediction]
    
      #print(predictions[IMAGE_INDEX])
      print("=====================")
      print(f"Max value (probability of prediction): {round(max_prediction*100)} %")
      #print(f"Sum: {sum_of_all_values}")
      print(f"Max index: {index_with_max_prediction}")
      print(f"Predicted label: {predicted_dog_breed} ")
      #return predicted_dog_breed
      print("=====================")
      return index_with_max_prediction
    
    def __createFilesNames(self, columnName, picturesNames, picturesPathDirectory = "", fileType = ".jpg"  ):
        filesNames = [picturesPathDirectory + fname + fileType for fname in picturesNames ]
        return filesNames
        
    def __preporcessImage(self,imagePath):
      '''
      Takes an image file path and turns the image into tensor.
      '''
      # Read an image file
      imgSize = self.__IMAGE_SIZE
      image = tf.io.read_file(imagePath) # reads a file and turns it into machine code string
      # turn the Jpeg into numerical tensor with 3 color channels (RGB between 0-255)
      image = tf.image.decode_jpeg(image,channels=3)
      # convert the color channel values (normalization) converts 0-255 to 0-1 values.
      # Normalization makes processing on the numbers more efficient.
      image = tf.image.convert_image_dtype(image,tf.float32)
      # Resize the image to (224,224)
      image = tf.image.resize(image,size=[imgSize,imgSize])
      # return Image
      return image

    def __loadPredictedImages(self,uniqueLabelsIndex):
        imagesFilesNames = []
        dogLabel = self.unique_values[uniqueLabelsIndex]
        
        for item in self.labels_csv :
            if (item[ self.__Y_ColumnName] == dogLabel):
                imagesFilesNames.append(item[self.__X_ColumnName])
            
        fullPaths = self.__createFilesNames(self.__Y_ColumnName, imagesFilesNames, self.__picturesPathDirectory , self.__files_type ) 
        
        for path in fullPaths :
            plt.imshow(path)
        
    def predict(self, picturePath):
        self.__model = self.__load_model(self.__modelPath)
        imageIndex = self.__predict_a_picture(picturePath, self.__IMAGE_SIZE, self.__model, self.__BATCH_SIZE)
        self.__loadPredictedImages(imageIndex)
