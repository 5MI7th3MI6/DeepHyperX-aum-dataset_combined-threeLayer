# %%
import numpy as np

#This function calculates the weigted median for a given data array and corresponding weights
#
# data:     nx1 array of elements
# weight:   nx1 array of weights for each element
def weightedMedian(data, weights):
    wSum = np.sum(weights)
    weights = weights / wSum  #make sure sum of weights is one
    
    ind = np.argsort(data) #sort elements increasing order
    weights = weights[ind] #sort weights
    data = data[ind]       #sort data
    
    weightSum = np.cumsum(weights) #find cumulative sum of weights
    
    j=0
    while(True):
       if(weightSum[j] >= 0.5):
           wMed = data[j]
           break
       j = j + 1

    return wMed           
    
#%%
def guidedMedianFilter(inputImg, hsiData):
    
    outputImg = np.zeros(np.shape(inputImg))
    predImage = np.pad(inputImg,((1,1),(1, 1)), 'edge')
    hsiData = np.pad(hsiData,((1,1),(1, 1),(0,0)), 'edge')
    
    rows = np.shape(predImage)[0]
    cols = np.shape(predImage)[1]
    
    for i in range(1,rows-2):
        for j in range(1,cols-2):
            predImgWindow = predImage[i-1:i+2,j-1:j+2]
            hsiDataWindow = hsiData[i-1:i+2,j-1:j+2,:]
      
            if(predImgWindow[1,1] != 0):
                predImgWindow[1,1] = 0
                
                hsiDataWindow2D = hsiDataWindow.reshape(9,200)
                predImgWindow2D = predImgWindow.reshape(9,)
                
                coeff = np.corrcoef(hsiDataWindow2D)[:,0]
                
                coeff = coeff[predImgWindow2D != 0]
                pImg = predImgWindow2D[predImgWindow2D != 0]

                if(np.sum(predImgWindow2D) == 0):
                    outputImg[i-1,j-1] = 0
                else:
                    outputImg[i-1,j-1] = weightedMedian(pImg,coeff)
                        
    return outputImg