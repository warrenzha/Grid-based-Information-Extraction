# Grid-based-Information-Extraction

## Authors: Wenyuan Zhao, Lindong Ye, Ziyan Cui

## Description
This project aims to solve the problem of key information extraction from visual rich document by a modified Grid-based mothod.

## Requirements
Run-time
Torch ==1.6, Torch ==1.1.1, Torchsummary ==1.5.1, TorchVision ==0.7.0  

## Running the Code
It contains two model files: CharGrid and BERTgrid.

Chargrid:  
(1) Run PRERProcessing.py 
Preprocess the input in the Data folder. There are three steps:  
1) Run new_preprocessing.py
-- Get input vectors that remove blank rows and columns  
2) Run new_preprocessing_bis.py
-- Get input vectors that remove similar rows and columns  
3) Run new_preprocessing_ter.py
-- Adjust to the fixed resolution, get the final one-hot vector input  

(2) Run the code of new_train for training  
(3) Run the new_test code to test

BERTgrid:  
(1) Run prerprocessing.py in PRERProcessing
Preprocess the input in the Data folder.  There are three steps:  
1) Run new_preprocessing.py
-- Get input vectors that remove blank rows and columns  
2) Run new_preprocessing_bis.py
-- Get an input vectors that remove similar rows and columns  
3) Run new_preprocessing_ter.py 
-- Adjust to fixed resolution, using the BERT pretraining model provided by Huggingface  

(2) Run the code of new_train for training  
(3) Run the new_test code to test

## Dataset
ICDAR 2019 SROIE.