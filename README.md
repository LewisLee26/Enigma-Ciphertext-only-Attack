# Enigma Ciphertext-only Attack

## Overview
This is a project for decrypting Enigma Machine messages using AI. This system achieves a decryption time of below 18 minutes on a single CPU. 

System is made up of two components, an enigma machine simulation and an AI model. The system iterates through the 1,054,560 possible rotor configurations. At each rotor configuration it decrypts the message and the model gives it a prediction. After completing all configurations, it takes the highest prediction and the settings that were used to get that prediction. 

## Enigma Machine simulation
The Enigma Machine simulation is the written in c. It is based on the Enigma I model. See the [Enigmac](https://github.com/LewisLee26/Enigmac) repository for more information.

## Model
The AI model is a trained like a binary classification model but within the system it acts like a regression model for predicting the most like text to be the decrypted text. 

### Datasets
The datasets are binary classfication datasets with text and a label of 1 for decrypted text and 0 for encrypted text. The model is trained on text that has been swapped with the Enigma Machine plugboard. This makes it so that it is more resiliant to high amounts of plugboard pairs unlike some other statistical methods. 

There are two types of datasets, a dataset with a range of plugboard pairs and datasets with a fixed number of plugboard pairs. The ranged plugboard dataset is used for training the model and the fixed plugboard datasets are only used for testing the models ability to identify that text with that specific number of plugsboard pairs. 
All of the datasets are created from the Bookcorpus dataset.

### Model architecture
The model is a one layer encoder-only transformer with 1 attention head and a hidden dimension of 512. 

Larger models have been tested but with only minor improvement. A finetuned DistilBERT achieves a significantly higher performance for less than 10 plugboards but with a much lower inference speed.

### Metrics

| Number of plugboard pairs | Accuracy | 
|---------------------------|------------|
| 0 | 91.2 |
| 2 | 91.5 |
| 4 | 90.9 |
| 6 | 90.3 |
| 8 | 87.9 |
| 10 | 85.0 |
| 12 | 84.8 |
| 13 | 79.7 |
| 0-13 | 79.9 |

| Model | Inference Speed (it/s) | 
|-------|-----------------|
| Base | 826.2 |
| Onnxruntime | 1062.7 |

