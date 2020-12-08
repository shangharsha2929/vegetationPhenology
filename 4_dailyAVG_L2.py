"""
Created on Sat Aug  1 13:12:23 2020
LEVEL-02 datasets creation
This python script computes daily average image and store it to a user defined path (i.e. average
images are exported to .jpg image in a folder called 'RGB'. Furthermore, it also computes daily 
GCC average as a grayscale image in a folder called 'GCC'. Folders are created in the source image
directory assigned by the user. 

Note: GCC images are to be divided by 255 pixelwise to get real GCC values.
@author: Shangharsha
"""
#################################################################################################
#Importing required modules
import glob
import os
import shutil
import cv2
import numpy as np
from datetime import datetime
from PIL import Image

#################################################################################################

#################################################################################################
#Get time now. This helps to compute total elapsed time for running the code.
#################################################################################################
start = datetime.now()

#Empty list to store the DOY information
doy = []

#Define the path for source images folder
imgSrc = r'E:\Internship\Final\Abisko\L1\2010'

#################################################################################################
#Automatically creating folders in the directory to save results into
#################################################################################################

#Try-except block is to pass overwrite directories if exists
folders = ['Temp','RGB', 'GCC', 'RCC']
for folder in folders:
    try:
        os.mkdir(os.path.join(imgSrc, folder))
    except:
        pass

#Path definition for intermediate file storage    
baseDst = imgSrc + '\Temp'

#################################################################################################
#################################################################################################

#1st Part
#Line of codes to automatically copy all valid images and store them in a folder named after DOY  
#Iterating all images
for img in sorted(glob.glob(os.path.join(imgSrc, '*.jpg'))):
    
    #Extracting image file name
    imgName = os.path.basename(img)
    
    #Day of Year information (DOY) extraction from image file name
    dayOfYear = int(imgName.split('_')[2])
    
    #Check if current DOY is in the list
    if dayOfYear not in doy:
        
        #Append the day of year in empty list DOY
        doy.append(dayOfYear)
        
        #Make a new folder in the given path with the 'doy' as folder name
        thePath = baseDst
        folders = [str(dayOfYear)]
        
        #Iterating the folders list to create each DOY as a new folder in given path
        for folder in folders:
            #Try-except block is to pass overwrite directories if exists
            try:
                os.mkdir(os.path.join(thePath, folder))
            except:
                pass
        
        #Copy the image from the source to destination folder
        imgDst = baseDst + '\\' + folders[0]
        #shutil.copy(img, imgDst)
    
    #If DOY exists in the doy list, copy the source image to the same folder
    shutil.copy(img, imgDst)
    
print ('\n')  
print ('Finished copying images to respective DOY folders.')
   
#################################################################################################
#################################################################################################

print ('\n')
print ('Starting to compute daily average images...')

#2nd part
#Line of codes to compute daily average from all valid images and save it as a .jpg file

#Save average image
imgSave = imgSrc + '\RGB'

for subdir in os.listdir(baseDst): 
    
    imgDir = baseDst + '\\' + subdir

    #Read all files in a directory as a numpy array
    #cv2.cvtColor for converting image from BGR to RGB
    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in glob.glob(os.path.join(imgDir, '*.jpg'))]
    
    #Compute element wise daily average
    avgImg = np.mean(images, axis = 0)
    
    #Converting float64 type ndarray to uint8
    intImage = np.around(avgImg).astype(np.uint8) #Round first and then convert to integer
    
    #Saving the daily average as image
    im = Image.fromarray(intImage)
    
    #Define path for saving image with given file name 
    saveDst = imgSave + '\\' + "_".join(os.listdir(imgDir)[0].split("_")[:3]) + '_RGB_L2_daily.jpg'
    
    #Save image in the defined path
    im.save(saveDst)
    
print ('\n')
print ('Daily averaged images are computed and stored successfully.')

#################################################################################################
#################################################################################################
    
#3rd part
#Line of code to delete pre-created DOY folders and files after finishing the processing  
shutil.rmtree(baseDst)

#################################################################################################
#################################################################################################

print ('\n')
print ('Starting to compute daily GCC and RCC images...')

#4th part
#Line of code to generate daily GCC and RCC as an image

#Save daily GCC and RCC image
gccSave = imgSrc + '\GCC'
rccSave = imgSrc + '\RCC'

#Iterating all daily average images to compute GCC and RCC on a pixel by pixel basis
for img in sorted(glob.glob(os.path.join(imgSave, '*.jpg'))):
    
    #Extracting image file name
    imgName = os.path.basename(img)

    #Reading image one by one
    cv_img = cv2.imread(img)
    
    #Extracting RGB bands as a separate numpy array
    B = cv_img[:,:,0]
    G = cv_img[:,:,1]
    R = cv_img[:,:,2]
      
    #Element wise addition of BGR array to calculate Total DN values in RGB band (i.e. R+G+B) 
    DNtotal = cv_img.sum(axis = 2)
    
    #Compute pixel wise GCC and RCC from daily average
    gcc = np.divide(G, DNtotal)
    rcc = np.divide(R, DNtotal)
    
    # Convert NAN to zero
    arr1 = np.nan_to_num(gcc, copy=False)
    arr2 = np.nan_to_num(rcc, copy=False)
    
    #Converting GCC and RCC to smoothly range from 0 - 255 as 'uint8' data type from 'float64'
    intImage1 = (arr1 * 255).astype(np.uint8) 
    intImage2 = (arr2 * 255).astype(np.uint8) 
    
    #Define path for saving image with given file name 
    saveDst1 = gccSave + '\\' + imgName.replace('RGB','GCC')
    saveDst2 = rccSave + '\\' + imgName.replace('RGB','RCC')
    
    #Save in the defined path as a Greyscale image
    cv2.imwrite(saveDst1, intImage1)  
    cv2.imwrite(saveDst2, intImage2) 
    
print ('\n')
print ('Daily averaged GCC and RCC images are computed and stored successfully.')
    
#################################################################################################
#Find out the total elapsed time and print out on the screen
#################################################################################################

end = datetime.now()
time_taken = end - start

print ('\n')
print ('Time elapsed: {}'.format(time_taken)) 

#################################################################################################
   