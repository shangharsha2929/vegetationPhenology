"""
Level-03 Dataset Creation
Created on Fri Feb 21 21:34:13 2020
This code extracts mean DN values over defined region of interest, use those mean DN values to
compute mean GCC, and RCC values over all images. Finally, from these values, daily average of
VIs are computed and both saved as well as plotted. 
Note: Assign the image path in 'thePath' variable. Before running the code, make sure all snow
covered images are moved to the folder named 'SnowImage' in the same path with the rest of images.
@author: Shangharsha
"""
#################################################################################################
#Module Declaration
#################################################################################################

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import pandas as pd
import datetime
from datetime import datetime as dt
from scipy import stats as s

#################################################################################################
#Get time now. This helps to compute total elapsed time for running the code.
#################################################################################################
start = dt.now()

#################################################################################################
#Empty lists to store the corresponding vegetation indices value
#################################################################################################

GCC = []
RCC = []
DOY = []

#Assign the file path for saving the result
thePath = r'E:\Internship\Final\Abisko\L1\2020'

#Initializing the empty dictionary to save the DOY as Key and GCC values from valid images
GCCdict1day = {}
RCCdict1day = {}
Reddict1day = {}
Grndict1day = {}
Bludict1day = {}
SnowdictTag = {}

#################################################################################################
#Display Region of Interest (ROI) Selection in the image  
#################################################################################################

#Random selection of one image from the image folder to show the extent of ROI
imgDir = thePath + '\\' + random.choice(os.listdir(thePath))

#Extract image name, station name, year
imName = os.path.basename(imgDir)
stnName = imName.split('_')[0]
yyyy = imName.split('_')[1][0:4]

#Loading image from the specified file
img = cv2.imread(imgDir)

#Multiple ROI definition for the image
pts1 = np.array([[1900, 180], [1400, 300], [2600, 300],[2800, 200]]) 
cv2.polylines(img, np.int32([pts1]), 1, (255, 0, 255), 7)

#################################################################################################
#Draw ROI on top of image to give visual representaion of ROI location 
#################################################################################################

#OpenCV represents image in reverse order BGR; so convert it to appear in RGB mode and plot it
plt.rcParams['figure.figsize'] = (16,8)
plt.figure(0)
plt.axis('on')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#################################################################################################
#Automatically creating folders in the directory to save results into
#################################################################################################

#Try-except block is to pass overwrite directories if exists
folders = ['Graph','CSV']
for folder in folders:
    try:
        os.mkdir(os.path.join(thePath, folder))
    except:
        pass

#################################################################################################
#Assigning path to create text file 
#################################################################################################        
pathVI = os.path.join(thePath + r'\CSV\VI_allImage.txt')

#Headers to be appended to the text file
heading = "Image DOY Red_ROI1 Green_ROI1 Blue_ROI1 Snow GCC_ROI1 RCC_ROI1"

#Open a file for writing the image name, corresponding DOY and vegetation indices
f1 = open(pathVI, 'w')

#Append header to the text file
f1.write(heading + "\n")

#################################################################################################
#Vegetation indices calculation within given ROI for all valid images
#################################################################################################

#Defining Region of Interest
#Create a mask with 0 as background having same size like that of original image
mask = np.zeros_like(img)
    
#Fill the polygon with white colour where we want to apply the mask
cv2.fillPoly(mask, np.int32([pts1]), (255,255,255))

#Iterating all images
for img in sorted(glob.glob(os.path.join(thePath, '*.jpg'))):

    #Reading image one by one
    cv_img = cv2.imread(img)
    
    #Extracting image file name
    imgName = os.path.basename(img)
    
    #Day of Year information (DOY) extraction from image file name
    dayOfYear = imgName.split('_')[2]
    DOY.append(dayOfYear)
    
    #Apply the mask and extract the image data within mask only
    masked = cv2.bitwise_and(cv_img, mask)
    
    #Splitting RGB image into separate bands
    B, G, R = cv2.split(masked)

    #Finding out mean DN of RGB bands within ROI 
    Rm = np.mean(np.ma.masked_equal(R, 0))
    Gm = np.mean(np.ma.masked_equal(G, 0))
    Bm = np.mean(np.ma.masked_equal(B, 0))

    #Total mean DN of ROI 
    TotalDN_ROI = Rm + Gm + Bm

    #Evaluation of visible band based vegetation indices
    #Green Chromatic Coordinate (GCC)
    g = round((Gm/(TotalDN_ROI)), 3)
    
    #Red chromatic Coordinate
    r = round((Rm/(TotalDN_ROI)), 3)    
    
    snow = 1 #Absence of snow
    
    #Appending GCC values for the images
    GCC.append(g)
    RCC.append(r)
    
    #Time series of vegetation indices saved as a text file in the given directory
    f1.write('{} {} {} {} {} {} {} {}\n'.format(imgName, dayOfYear, Rm, Gm, Bm, snow, g, r))
    
    #Update dictionary with DOY and its associated multiple vegetation indices values
    if dayOfYear in GCCdict1day:
        GCCdict1day[dayOfYear].append(g)
        RCCdict1day[dayOfYear].append(r)
        Reddict1day[dayOfYear].append(Rm)
        Grndict1day[dayOfYear].append(Gm)
        Bludict1day[dayOfYear].append(Bm)
        SnowdictTag[dayOfYear].append(snow)
        
    else:
        GCCdict1day[dayOfYear] = [g]
        RCCdict1day[dayOfYear] = [r]
        Reddict1day[dayOfYear] = [Rm]
        Grndict1day[dayOfYear] = [Gm]
        Bludict1day[dayOfYear] = [Bm]
        SnowdictTag[dayOfYear] = [snow]

################################################################################################
#If there is snow covered images in the folder path, comment lines of codes from line number 183
#to line number 251.
################################################################################################        

#Empty list to store snow covered image GCC and its respective DOY
doySnow = []
gccSnow = []
rccSnow = []

#Folder path definition of snow covered images        
snowImg = thePath + '\SnowyImage'

#Iterating through snow covered images
for img in sorted(glob.glob(os.path.join(snowImg, '*.jpg'))):

    #Reading image one by one
    cv_img = cv2.imread(img)
    
    #Extracting image file name
    imgName = os.path.basename(img)
    
    #Day of Year information (DOY) extraction from image file name
    dayOfYear = imgName.split('_')[2]
    DOY.append(dayOfYear)
    
    #Apply the mask and extract the image data within mask only
    masked = cv2.bitwise_and(cv_img, mask)
    
    #Splitting RGB image into separate bands
    B, G, R = cv2.split(masked)

    #Finding out mean DN of RGB bands within ROI 
    Rm = np.mean(np.ma.masked_equal(R, 0))
    Gm = np.mean(np.ma.masked_equal(G, 0))
    Bm = np.mean(np.ma.masked_equal(B, 0))

    #Total mean DN of ROI 
    TotalDN_ROI = Rm + Gm + Bm

    #Evaluation of visible band based vegetation indices
    #Green Chromatic Coordinate (GCC)
    g = round((Gm/(TotalDN_ROI)), 3)
    
    #Red chromatic Coordinate
    r = round((Rm/(TotalDN_ROI)), 3)
    
    snow = 0 #Presence of snow
    doySnow.append(dayOfYear)
    gccSnow.append(g)
    rccSnow.append(r)
       
    #Appending GCC and RCC values for the images
    GCC.append(g)
    RCC.append(r)
    
    #Time series of vegetation indices saved as a text file in the given directory
    f1.write('{} {} {} {} {} {} {} {}\n'.format(imgName, dayOfYear, Rm, Gm, Bm, snow, g, r))
    
    #Update dictionary with DOY and its associated multiple vegetation indices values
    if dayOfYear in GCCdict1day:
        GCCdict1day[dayOfYear].append(g)
        RCCdict1day[dayOfYear].append(r)
        Reddict1day[dayOfYear].append(Rm)
        Grndict1day[dayOfYear].append(Gm)
        Bludict1day[dayOfYear].append(Bm)
        SnowdictTag[dayOfYear].append(snow)
        
    else:
        GCCdict1day[dayOfYear] = [g]
        RCCdict1day[dayOfYear] = [r]
        Reddict1day[dayOfYear] = [Rm]
        Grndict1day[dayOfYear] = [Gm]
        Bludict1day[dayOfYear] = [Bm]
        SnowdictTag[dayOfYear] = [snow]        

#################################################################################################
#Close the file when done 
#################################################################################################
f1.close()

#################################################################################################
#Export .txt file as .csv file with all the information   
#################################################################################################

#Read the .txt file as a dataframe 
df = pd.read_table(pathVI, delim_whitespace = True)

#Sort the dataframe in increasing DOY order
sortedDF = df.sort_values('Image')

#Export the dataframe as a .csv file
fileName = os.path.join(thePath + r'\CSV\{}_{}_allImages.csv'.format(stnName, yyyy))
sortedDF.to_csv(fileName, index=False)

###################################################################################################
#Finding mean vegetation indices values from all valid images within a given DOY
###################################################################################################
        
#Dictionaries to store mean indices per day from valid images (Excluding no data values for a DOY)
avgGCC = {}
avgRCC = {}
avgR = {}
avgG = {}
avgB = {}
sTag = {}

#Assigning path to create new text file for storing daily averaged indices values
path_avgGCC = os.path.join(thePath + r'\CSV\avgGCC1Day.txt')

#Headers to be appended to the text file
header1 = "TIMESTAMP DOY Snow Red_ROI1 Green_ROI1 Blue_ROI1 GCC_ROI1 RCC_ROI1"
header2 = "YYYY-MM-DD None None DN DN DN Fraction Fraction"
header3 = "None None None AVG AVG AVG AVG AVG"

#Open a file for writing the corresponding DOY and vegetation indices
f4 = open(path_avgGCC, 'w')

#Append header to the text file
f4.write(header1 + "\n")
f4.write(header2 + "\n")
f4.write(header3 + "\n")

#Iterating over all dictionary keys, value pairs and average the items
for (k, v), (k1, v1), (k2, v2), (k3, v3), (k4, v4), (k5, v5) in zip(sorted(GCCdict1day.items()), \
    sorted(RCCdict1day.items()), sorted(Reddict1day.items()), sorted(Grndict1day.items()), sorted(Bludict1day.items()), sorted(SnowdictTag.items())):
    
    #val, val2 is the lists of GCC, & RCC values of all valid images on that DOY
    avgGCC[k] = round(sum(v)/len(v), 3)
    avgRCC[k1] = round(sum(v1)/len(v1), 3)
    avgR[k2] = round(sum(v2)/len(v2), 3)
    avgG[k3] = round(sum(v3)/len(v3), 3)
    avgB[k4] = round(sum(v4)/len(v4), 3) 
    sTag[k5] = int(s.mode(v5)[0])
    
    #Extracting timestamp information from day of year and year for the dataset
    yyyy_doy = yyyy + '+' + k
    timeStamp = datetime.datetime.strptime(yyyy_doy, "%Y+%j").strftime('%Y-%m-%d')
    
    #Time series of daily average VIs saved as a text file in the given directory
    f4.write('{} {} {} {} {} {} {} {}\n'.format(timeStamp, k, sTag[k5], avgR[k2], avgG[k3], avgB[k4], avgGCC[k], avgRCC[k1]))

#Close file
f4.close()

#################################################################################################
#Export .txt file as .csv file with all the information   
#################################################################################################

#Read the .txt file as a dataframe 
df = pd.read_table(path_avgGCC, delim_whitespace=True)

#Export the dataframe as a .csv file
fileName = os.path.join(thePath + r'\CSV\{}_{}_L3_daily.csv'.format(stnName, yyyy))
df.to_csv(fileName, index=False)

#################################################################################################
#Remove all .txt file in the directory   
#################################################################################################
dir_name = os.path.join(thePath + r'\CSV')
test = os.listdir(dir_name)

for item in test:
    if item.endswith(".txt"):
        os.remove(os.path.join(dir_name, item))
     
#################################################################################################
#Time series of daily averaged vegetation indices plotted against corresponding DOY   
#################################################################################################

#Plotting time series of GCC vegetation index
plt.figure(1)
plt.rcParams['figure.figsize'] = (16,8)
plt.plot([int(i) for i in DOY], GCC, 'o', color = 'grey', markersize = 4, alpha = 0.1, label = 'All image GCC')
plt.plot([int(i) for i in doySnow], gccSnow, 'o', color = 'cornflowerblue', markersize = 4, alpha = 0.5, label = 'Snowy image GCC') 
plt.plot([int(j) for j in sorted(avgGCC.keys())], [avgGCC[x] for x in sorted(avgGCC.keys())], 
         'r^', markersize = 6, mfc = 'none', label = 'Daily Average')
plt.xticks(range(0, 365, 10), rotation = 45, fontsize = 16)
plt.yticks(fontsize = 16) 
plt.grid(True, alpha = 0.3)
plt.xlabel('Day of Year (DOY)', fontsize = 20)
plt.ylabel('Green Chromatic Coordinate (GCC)', fontsize = 20)
plt.legend(loc = 'upper left', fontsize = 18)
plt.savefig(os.path.join(thePath + r'\Graph\GCC_1Day.jpg'))

#Plotting time series of RCC vegetation index
plt.figure(2)
plt.rcParams['figure.figsize'] = (16,8)
plt.plot([int(i) for i in DOY], RCC, 'o', color = 'grey', markersize = 4, alpha = 0.1, label = 'All image RCC')
plt.plot([int(i) for i in doySnow], rccSnow, 'o', color = 'cornflowerblue', markersize = 4, alpha = 0.5, label = 'Snowy image RCC')
plt.plot([int(j) for j in sorted(avgRCC.keys())], [avgRCC[x] for x in sorted(avgRCC.keys())], 
         'ro', markersize = 6, mfc = 'none', label = 'Daily Average')
plt.xticks(range(0, 365, 10), rotation = 45, fontsize = 16)
plt.yticks(fontsize = 16) 
plt.grid(True, alpha = 0.3)
plt.xlabel('Day of Year (DOY)', fontsize = 20)
plt.ylabel('Red Chromatic Coordinate (RCC)', fontsize = 20)
plt.legend(loc = 'upper left', fontsize = 18)
plt.savefig(os.path.join(thePath + r'\Graph\RCC_1Day.jpg'))

#################################################################################################
#Find out the total elapsed time and print out on the screen
#################################################################################################

end = dt.now()
time_taken = end - start

#These line of codes will print out the total elapsed time
print ('\n')
print ('Time elapsed: {}'.format(time_taken))
print ('\n')      
print ('Successfully computed daily average of GCC and RCC. Check .csv file in the defined path.')
#################################################################################################