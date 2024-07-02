# Dermatological Image Classification

This repository contains the implementation of a dermatological image classifier that differentiates between melanoma and melanocytic nevus skin lesions. This project was developed as part of a supervised machine learning assignment at Chalmers University of Technology.

## Table of Contents
- [Introduction](#introduction)
- [Technical Solutions](#technical-solutions)
  - [Data Augmentation](#data-augmentation)
  - [Data Loading](#data-loading)
  - [Model Preparation, Loss Function, and Optimizer](#model-preparation-loss-function-and-optimizer)
- [Experiment](#experiment)
  - [Training, Validation Function, and Training Execution](#training-validation-function-and-training-execution)
  - [Test Data Evaluation](#test-data-evaluation)
- [Results](#results)
  - [Training and Evaluation](#training-and-evaluation)
  - [Evaluation on Test Set](#evaluation-on-test-set)
  - [Comparison of Results With and Without Data Augmentation](#comparison-of-results-with-and-without-data-augmentation)
- [Conclusion](#conclusion)
- [Limitations](#limitations)
- [Ethical Considerations](#ethical-considerations)
  - [Data Privacy](#data-privacy)
  - [Bias in Training Data](#bias-in-training-data)
  - [Possibility of Misclassifications on Patient Outcomes](#possibility-of-misclassifications-on-patient-outcomes)
- [License](#license)

## Introduction

Skin cancer is one of the most common malignancies worldwide, with melanoma being particularly dangerous if not detected early. This project aims to develop a machine learning model to classify skin lesions as either melanoma or melanocytic nevus using the 2018 ISIC challenge dataset.

## Technical Solutions

### Data Augmentation

We applied various transformations using PyTorch's `torchvision.transforms` module to preprocess the training and validation datasets. This introduced randomness and variability in the training process to improve the model's robustness.

### Data Loading

Images were loaded in batches of 4 to efficiently manage memory usage and computation. Shuffling at each epoch was implemented to reduce overfitting.

### Model Preparation, Loss Function, and Optimizer

We used the ResNet model pre-trained on ImageNet for its stability and feature extraction capabilities. The loss function used was CrossEntropyLoss, and the optimizer was SGD. A step-based learning rate scheduler was also employed.

## Experiment

### Training, Validation Function, and Training Execution

Model training was conducted over 25 epochs using a function designed to manage training and validation phases, learning rate modifications, forward passes, loss computation, and parameter updates.

### Test Data Evaluation

The model was evaluated on an unseen test dataset processed through the same preprocessing pipeline as the training data. The accuracy was calculated by comparing predicted labels against true labels.

## Results

### Training and Evaluation

The model showed consistent improvement in both training and validation accuracy over 25 epochs. The highest validation accuracy achieved was 89.616% at epoch 24.

### Evaluation on Test Set

The model achieved an accuracy of 88.5% on the test dataset, demonstrating strong generalization capabilities.

### Comparison of Results With and Without Data Augmentation

Data augmentation significantly improved the model's performance, with a test accuracy of 88.5% compared to 50.66% without augmentation.

## Conclusion

Our model successfully distinguishes between melanoma and melanocytic skin lesions, thanks to data augmentation and the use of a pre-trained ResNet model. 

## Limitations

The study is limited by the dataset size and the simplified classification task of only distinguishing between melanoma and nevus. This may affect the model's generalizability and effectiveness in real-world contexts.

## Ethical Considerations

### Data Privacy

Sensitive patient data must be anonymized and securely stored, with explicit patient permission policies and regulatory frameworks in place to ensure compliance.

### Bias in Training Data

Training data should represent diverse demographics to reduce the risk of biased algorithms that misdiagnose certain groups.

### Possibility of Misclassifications on Patient Outcomes

Automated diagnosis systems should enhance rather than replace human competence, given the possibility of misclassification and its impact on patient outcomes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
