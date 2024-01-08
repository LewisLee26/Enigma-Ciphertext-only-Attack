# Enigma Ciphertext-only Attack

## Overview
This is a project for decrypting Enigma Machine messages using AI. This system is able to decrypt messages in less than _ minutes on a single CPU. 

System is made up of two components, an enigma machine simulation and an AI model. The system iterates through the 1,054,560 possible rotor configurations. At each rotor configuration it decrypts the message and the model gives it a prediction. After completing all configurations, it takes the highest prediction and the settings that were used to get that prediction. 

## Enigma Machine simulation
The Enigma Machine simulation is the written in C. It is based on the Enigma I model. See the [Enigmac](https://github.com/LewisLee26/Enigmac) repository for more information.

## Model
The AI model is a trained like a binary classification model but within the system it acts like a regression model for predicting the most like text to be the decrypted text. 

### Datasets
The datasets are binary classfication datasets with text and a label of 1 for decrypted text and 0 for encrypted text. The model is trained on text that has been swapped with the Enigma Machine plugboard. This makes it so that it is more resiliant to high amounts of plugboard pairs unlike some other statistical methods. 

There are two types of datasets, a dataset with a range of plugboard pairs and datasets with a fixed number of plugboard pairs. The ranged plugboard dataset is used for training the model and the fixed plugboard datasets are only used for testing the models ability to identify that text with that specific number of plugsboard pairs. 

All of the datasets are created from the Wikipedia 20220301.en dataset.

### Model architecture
The model is a one layer encoder-only transformer with 3 attention head and a hidden dimension of 512. The model uses a one hot embedding for it's character level input with the tokens A-Z represented as 1-26 and 0 as an padding token. The model has 3 attention heads to maximise the number of attention heads without increasing the embedding dimension. 

Although the model is trained on a binary classification dataset, the model doesn't include a sigmoid function for the output layer. Instead, the sigmoid is applied outside the model during training. The hope is that this allows for the model to act similar to a regression model during deployment and allow for greater accuracy than with a sigmoid function.

Larger models have been tested but with only minor improvement. A finetuned DistilBERT achieves a significantly higher performance for but with a much lower inference speed.

### Metrics

**Accuracy**
| Number of plugboard pairs | Base | Onnxruntime | 
|---------------------------|------|-------------|
| 0 | 99.6 | 99.7 |
| 2 | 99.8 | 99.9 |
| 4 | 99.3 | 99.3 |
| 6 | 97.2 | 97.1 |
| 8 | 92.8 | 92.6 |
| 10 | 85.8 | 85.5 |
| 12 | 81.1 | 80.8 |
| 13 | 78.3 | 78.0 |
| 0-13 | 78.9 | 78.6 |

*This table shows the accuracy on a binary classification test dataset and not deployment*

**Inference Speed**
| Model | Inference Speed (it/s) | 
|-------|-----------------|
| Base | 388.1 |
| Onnxruntime | 1123.4 |

*This table shows the model's inference speed on a test dataset and not deployment* 


