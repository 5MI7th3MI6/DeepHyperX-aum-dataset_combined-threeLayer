# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:24:42 2022

@author: sdinc
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm

class threeLayerHSIClassification(BaseEstimator, ClassifierMixin):
    """Fits a logistic regression model on tree embeddings.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scaler = MinMaxScaler(feature_range=(0, 0.95),clip=True)

    
    def fit(self, X, y):
        numOfSamples, numOfFeatures = np.shape(X)
        self.numOfClasses = np.unique(y).shape[0]
        
        X_scaled = self.scaler.fit_transform(X)
        
        #There are 16 contours, each has 100x200 matrix.
        #100 is number of bins in histogram. And 200 is number of features.
        numOfContourBins = 100
        self.allContours = np.zeros((self.numOfClasses,numOfContourBins,numOfFeatures))
        
        #There are 16 references. Each has 200x1 vector.
        self.allReferences = np.zeros((self.numOfClasses,numOfFeatures))
        
        for classLabel in range(self.numOfClasses):
            P_c = X_scaled[(y == classLabel+1),:] #current class training samples
            P_r = X[(y == classLabel+1),:] #current class training samples
            
            #find the contour
            contour = self.findContour(P_c,numOfFeatures)
            self.allContours[classLabel] = contour
            
            #find the reference
            reference = np.percentile(P_r, 50, axis=0)
            self.allReferences[classLabel] = reference
    
    
    #% This function transforms a sample using contours and references
    def transform(self, X, y=None):
        numOfSamples, numOfFeatures = np.shape(X)
        
        X_transformed = np.zeros((numOfSamples, 2*self.numOfClasses))
        X_scaled = self.scaler.transform(X)
    
        #this loops transforms every sample. Finds new features from contour and reference learners.
        for tsInd in tqdm(range(0,numOfSamples), desc ="Transforming Samples"):
            P_c = X_scaled[tsInd,:]
            P_r = X[tsInd,:]
            
            contour_scores = np.zeros((self.numOfClasses))
            referrence_scores = np.zeros((self.numOfClasses))
            
            for classLabel in range(self.numOfClasses):
                contour = self.allContours[classLabel]
                contour_scores[classLabel] = self.findContourScore(P_c, contour,numOfFeatures)
                
                reference = self.allReferences[classLabel]
                referrence_scores[classLabel] = self.findReferenceScore(P_r, reference,numOfFeatures)
            
            X_transformed[tsInd,:] = np.hstack((contour_scores,referrence_scores))
        
        return X_transformed
        
    
    #% This function finds the contour of each class
    def findContour(self,P_c,numOfFeatures):
        contour = np.zeros(shape=(100,numOfFeatures))
        for j in range(numOfFeatures):
            f_j = P_c[:,j]
            hist, bin_edges = np.histogram(f_j, bins=np.arange(0,1.01,0.01), density=True)
            contour[:,j] = hist
        return contour
    
    #% This function calculates contour score of a test sample
    def findContourScore(self,testSample, contour,numOfFeatures):
        ind = np.digitize(testSample,np.arange(0,1.01,0.01)) 
        p = contour[ind,range(numOfFeatures)]
                
        return np.sum(p) 
    
    #% This function calculates contour score of a test sample
    def findReferenceScore(self,testSample, reference,numOfFeatures):
        c = np.corrcoef(testSample, reference)[0, 1]
        return max(0.0, c)