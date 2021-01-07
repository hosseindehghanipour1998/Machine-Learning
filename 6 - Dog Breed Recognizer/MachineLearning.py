# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:16:59 2021

@author: Hossein
"""
# ===== Import necessary Tools =====
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub 
from IPython.display import Image
import os
import matplotlib.pyplot as plt
import datetime

# Check Version and Availability

# Import Files labels


# Visualize the Data set





class MyMachine:
    

    
    # -------------- Public Methods ----------------------
    def __checkVersions(self):
        print("TF Version :",tf.__version__) # Prints the Version of Tensor Flow we have installed on our system.
        print("TF Hub Version :",hub.__version__)
        print("GPU : " , "Available" if tf.config.list_physical_devices("GPU") else "Not Available")

    

        
    def showLabelsCSV(self):
        print(self.labels_csv.describe())
        self.labels_csv.head()
        
    
    def showImage(ImagePath):
        '''
        

        Parameters 
        ----------
        ImagePath : TYPE
            DESCRIPTION.
        
        Example : /train/Picture.jpg
        
        Returns 
        -------
        None.
        
        '''
        Image(ImagePath)
     
    def visualizeLabelsCSV(self, columnName):
        self.labels_csv[columnName].value_counts().plot.bar(figsize=(20,10)) 
        return
    
        
    
    def __createFilesNames(self, columnName, picturesPathDirectory = "", fileType = ".jpg"):
        '''
        Parameters
        ----------
        prefix : TYPE, optional 
             The picturesPathDirectory is the Path of the pictures dataset
             The default is "".
             
        filesLabels : TYPE, optional
            The file that has the labels of the dataset inside it.
            DESCRIPTION. The default is self.labels_csv.
            
        columnName : TYPE
            Name of the column in the labels that contain the name of the pictures.
            Ex: breed, id
            
            
        fileType : TYPE, optional
            Type of the Pictures: JPEG, JPG, ...
            The default is ".jpg".

        Returns
        -------
        None.

        '''
        self.filesNames = [picturesPathDirectory + fname + fileType for fname in self.labels_csv[columnName]]    
        
        
    def __checkFilesNumbersMatch(self):
        
        if ( len(os.listdir(self.__trainPicturesPath)) == len(self.filesNames) ) :
            print("Train-Set images Match !")
        else :
            print("rain-Set images Don't Match !")
        
        
    def __fetchImagesNames(self, trainingSetImagesPath, testSetImagesPath = ""):
        
        for image in os.listdir(trainingSetImagesPath):
            self.trainingImagesNames.append(image)
            
        if (testSetImagesPath != "" ):
            for image in os.listdir(testSetImagesPath):
                self.testImagesNames.append(image)
        
    def __createAndSplitDataSets(self, Y_CoulumnName, trainigImagesNo, testSize=0.2, randomState = 42):
        '''
        

        Parameters
        ----------
        Y_CoulumnName : TYPE
            Name of the Column that holds Y values.
        trainigImagesNo : TYPE
            Number of Images we want our model to learn by.
        testSize : TYPE, optional
            Test Size Split. The default is 0.2.
        randomState : TYPE, optional
            Random Seed. The default is 42.

        Returns
        -------
        x_train : TYPE

        x_val : TYPE

        y_train : TYPE

        y_val : TYPE


        '''
        labels = self.labels_csv[Y_CoulumnName].to_numpy() 
        unique_values = np.unique(labels)
        self.__unique_values = unique_values
        boolean_labels = [label == unique_values for label in labels]
        
        self.X = self.filesNames
        self.y = boolean_labels
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(self.X[:trainigImagesNo],self.y[:trainigImagesNo],test_size = testSize,random_state=randomState)
        return x_train, x_val, y_train, y_val
        
    def getImageShape(self, imageIndex = 0 ):
        from matplotlib.pyplot import imread
        image = imread(self.filesNames[42])
        return image.shape
    
    # we will talk about this number later. 224*224

    def __preporcessImage(self,imagePath):
      '''
      Takes an image file path and turns the image into tensor.
      '''
      # Read an image file
      imgSize = self.IMAGE_SIZE
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
        
    def __get_image_label(self,image_path, label):

        image = self.__preporcessImage(image_path)
        return image,label    
  
    
      
    def __create_data_batches(self, X, batch_size, y=None, valid_data=False, test_data = False):
      """
      1 - Creates batches out of data
      2 - shuffles the data but doesn't shuffle if it is validation data. Only train data should be shuffled
      3 - also accepts test data as input and makes batches out of it.
      """
      # If 
      if (test_data):
        print("Creating Test Batches")
        
        data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
        
        data_batch = data.map(self.__preporcessImage).batch(batch_size)
        return data_batch
        
      elif (valid_data):
        print("Creating Validation Data Batch")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        data_batch = data.map(self.__get_image_label).batch(batch_size)
        return data_batch
    
      else: # A training Set 
        print("Createing a Training Set Batch")
        # Turn filePaths and Labels into Tensors
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
        # Shuffling the pathNames and Labels before mapping the image_process()
        # we shuffle before procesign because shuffling the image takes a longer time
        print
        data = data.shuffle(buffer_size=len(X))
        
        data_batch = data.map(self.__get_image_label).batch(batch_size)
        return data_batch
    
    


    # Create a function for viewing images in a data batch

    def __show_25_images(self, images, labels):
      """
      Displays a plot of 25 images and their labels from a data batch.
      """
      # Setup the figure
      plt.figure(figsize=(10, 10))
      # Loop through 25 (for displaying 25 images)
      for i in range(25):
        # Create subplots (5 rows, 5 columns)
        ax = plt.subplot(5, 5, i+1)
        # Display an image 
        plt.imshow(images[i])
        # Add the image label as the title
        plt.title(self.__unique_values[labels[i].argmax()])
        # Turn the grid lines off
        plt.axis("off")

        
    # Create a function which builds a Keras model
    
    def __create_model(self, inputShape, output_shape, model_url):
      print("Building model with:", model_url)
    
      # Setup the model layers
      model = tf.keras.Sequential([
        hub.KerasLayer(model_url), # Layer 1 (input layer)
        # gets our pre-defined model and uses it as the first layer
        tf.keras.layers.Dense(units=output_shape, activation="softmax") # Layer 2 (output layer)
        # For binary classification the Activation is Sigmoid but for Multi-Class classification its Softmax.
      ])
    
      # Compile the model
      model.compile(
          # Compiling is like going at the bottom of a hill blind-folded.
          loss=tf.keras.losses.CategoricalCrossentropy(), # How well our model is geussing
          optimizer=tf.keras.optimizers.Adam(), # How to guess to be the most optimized.
          # Adam optimizer tells us how to descend in order to be most optimized path.
          metrics=["accuracy"] # Evaluating the guesses after it has learned.
          # The mectrics tell us how well we have done or how accurate we are.
      )
    
      # Build the model
      model.build(inputShape)
    
      return model

    def getModelSummary(self):
        try:
            if(self.__model != None):
                print("Model Summary:")
                self.__model.summary()
        except:
            print("No models Detected.")



    # Create a function to build a TensorBoard callback
    def __create_tensorboard_callback(self,callbackLogsPath):
      '''
      This function creates a call-back, which will help us visualaize and log the process of learning by our model.
      '''
      # Create a log directory for storing TensorBoard logs
      # before running the next line of code, create a folder named Log in any directory you want and then pass the path of that directory to the fucntion.
      logdir = os.path.join(callbackLogsPath,
                            # Make it so the logs get tracked whenever we run an experiment
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
      return tf.keras.callbacks.TensorBoard(logdir)  
  
    def __train_model(self, earlyStopping, inputShape, outputShape, modelURL, epochs_num, callBackLogsPath ):
      """
      Trains a given model and returns the trained version.
      """
      # Create a model
      model = self.__create_model(inputShape, outputShape, modelURL) # We have already implemented it.
    
      # Create new TensorBoard session everytime we train a model
      tensorboard = self.__create_tensorboard_callback(callBackLogsPath) # we have already implemented it.
    
      # Fit the model to the data passing it the callbacks we created
      
      # Running this piece of code on GPU instead of CPU
      from numba import jit, cuda
      #jit(target ="cuda")
      from timeit import default_timer as timer
      start = timer()
      
      print(self.__train_data)
      model.fit(x=self.__train_data,
                epochs=epochs_num,
                validation_data=self.__val_data,
                validation_freq=1, # how often we want to test the patterns? how many epochs we are validating?
                callbacks=[tensorboard, earlyStopping]) # tensorboard and early_stopping are defined by us.
      
      
      print(f"Training Time: {timer()-start}")
      # Return the fitted model
      return model         
  
    
    def initializer(self,
                              trainCoulumnName,
                              valuesColumnName,
                              filesType, 
                              TrainingPicturesPathDirectory, 
                              TestingPicturesPathDirectoy,
                              trainingImagesNo,
                              callbackLogsPath,
                              modelsSavingPath,
                              testPercentage = 0.2,
                              randomState = 42
                              
                              ):
        
        self.__trainCoulumnName = trainCoulumnName
        self.__valuesCoulumnName = valuesColumnName
        self.__filesType = filesType
        self.__trainPicturesPath = TrainingPicturesPathDirectory
        self.__testPicturesPath = TestingPicturesPathDirectoy
        self.__training_images_number = trainingImagesNo
        self.__testPercentage = testPercentage
        self.__randomState = randomState
        self.__callbackLogsPath = callbackLogsPath
        self.__modelsSavingPath = modelsSavingPath
        self.__InitializedCalled = True    
        return 0
        
    def __save_model(self,model, savedModelsPathDir, modelName=None):
      """
      Saves a given model in a models directory and appends a suffix (string).
      """
      # Create a model directory pathname with current time
      modeldir = "" #os.path.join(savedModelsPathDir,datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
      model_path = savedModelsPathDir + "" + modelName + ".h5" # save format of model
      print(f"Saving model to: {model_path}...")
      model.save(model_path)
      print(f"Model Saved in: {model_path}")
      return model_path   
    
    def __init__(self, CSVPath):
        
        
        # Global Vars
        self.IMAGE_SIZE = 224
        self.BATCH_SIZE = 32
        self.MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
        self.NUM_EPOCHS = 100
        
        self.labels_csv = pd.read_csv(CSVPath)
        self.__train_data = None
        self.__val_data = None
        
        # ----------------------- Initializer ------------
        self.__trainCoulumnName = None
        self.__valuesCoulumnName = None
        self.__filesType = None
        self.__trainPicturesPath = None
        self.__testPicturesPath = None
        self.__training_images_number = None
        self.__testPercentage = None
        self.__randomState = None
        self.__callbackLogsPath = None
        self.__modelsSavingPath = None

        # ----------------- Inside Methods ---------------
        self.X = None
        self.y = None
        self.filesNames = []
        self.testImagesNames = []
        self.trainingImagesNames = []
        self.__unique_values = []
        self.__model = None
        # -------------------- Sequence Control Variables
        self.__InitializedCalled = False
    
    def Train(self):
        if(self.__InitializedCalled):
            print("=========== VERSION ====================")
            self.__checkVersions()
            
            print("=========== PRE-PROCESSING ====================")
            self.__createFilesNames(self.__trainCoulumnName , self.__trainPicturesPath, self.__filesType ) #Fills: self.filesNames 
            self.__checkFilesNumbersMatch()
            self.__fetchImagesNames( self.__trainPicturesPath, self.__testPicturesPath)#Fills: self.testImagesNames, self.trainingImagesNames
            x_train, x_val, y_train, y_val = self.__createAndSplitDataSets( self.__valuesCoulumnName, self.__training_images_number, self.__testPercentage, self.__randomState)
            #Fills: self.__unique_values, self.X, self.y
            self.__train_data = self.__create_data_batches(x_train,self.BATCH_SIZE,y_train)
            self.__val_data = self.__create_data_batches(x_val,self.BATCH_SIZE,y_val,valid_data=True)
            
            print("=========== VISUALIZING DATA ====================")
            ## Visualize the Training and Validation Images
            train_images, train_labels = next(self.__train_data.as_numpy_iterator())
            self.__show_25_images(train_images, train_labels)
            val_images, val_labels = next( self.__val_data.as_numpy_iterator())
            self.__show_25_images(val_images, val_labels)
            
            INPUT_SHAPE = [None, self.IMAGE_SIZE, self.IMAGE_SIZE , 3]
            OUTPUT_SHAPE = len(self.__unique_values)

            self.__create_tensorboard_callback(self.__callbackLogsPath)
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
            
            print("=========== TRAINING  ====================")
            print("Training has Started...")

            self.__model = self.__train_model( early_stopping,  INPUT_SHAPE, OUTPUT_SHAPE, self.MODEL_URL,  self.NUM_EPOCHS, self.__callbackLogsPath)
            self.getModelSummary() #Prints model's summary information.
            print("Training Completed")
            
            print("=========== EVALUATING ====================")
            print(f"Evaluation : {self.__model}")
            results = ("NA","NA")#self.__model.evaluate(x_val, y_val, self.BATCH_SIZE)
            print(f"Loss: {results[0]} | Accuracy: {results[1]}")
            
            print("=========== SAVING MODEL ====================")
            modelName = input("Model Name : ")
            evaluationFactors = "_LOSS_" + str(results[0]) + "_ACCURACY_" + str(results[1]) + "_IMG_NO_" + str(self.__training_images_number)
            modelName = str(modelName) + evaluationFactors
            self.__save_model(self.__model,self.__modelsSavingPath,modelName)
        else:
            print("You must Call the Initializer First.")
    

        
        