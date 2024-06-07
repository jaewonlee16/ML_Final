# Final Project - Typo Correction for RGB Character Image Sequence

## Overview

This project focuses on correcting typos in words represented as sequences of RGB character images. Given a sequence of character images of length \( T \), the goal is to output the correct word sequence in character form. The input consists of a 3-channel RGB-scale character image sequence, and the output is a 1-dimensional integer sequence corresponding to the alphabet characters.

- **Input:** RGB-scale character image sequence of shape (𝑇, 28, 28, 3), where 4 ≤ 𝑇 ≤ 10
- **Output:** 1-dimensional integer sequence of length 𝑇, where each integer corresponds to an alphabet character (case insensitive)

## Architecture

The model utilizes a Seq2Seq architecture, combining a CNN + RNN Encoder and an RNN Decoder.

- **Seq2Seq Model:** CNN + RNN Encoder and RNN Decoder
- **Decoder Output:** Logit corresponding to the number of classes (fixed at 28 in the final project)
  - Alphabet: 26 characters
  - Start token (`<s>`): 1 token
  - Padding token (`<pad>`): 1 token

### Encoder

- **CNN:** Custom CNN class with freely defined parameters and hyper-parameters
- **RNN:** Can choose any RNN model (LSTM, GRU, etc.)
- **Additional Parameters:** Defined freely and used in the forward function

### Decoder

- **RNN:** Can choose any RNN model (LSTM, GRU, etc.)
- **Additional Parameters:** Defined freely and used in the forward function

### Training and Evaluation

- **Training Set:** 67,890 various length (4 ≤ 𝑇 ≤ 8) image sequences
- **Validation Set:** 9,670 various length (4 ≤ 𝑇 ≤ 8) image sequences
  - Consists of words not present in the training set
- **Challenge Set:** 8,714 various length (9 ≤ 𝑇 ≤ 10) image sequences
  - Consists of words and lengths not present in the training set

## Tasks

### Normal Task

- Train the model using the training set and evaluate using the validation set
- Absolute performance evaluation based on the validation set

### Challenge Task

- Train the model using the training set and evaluate using the challenge set
- Relative performance evaluation based on the challenge set

## Search Space and Constraints

- No additional data usage beyond the provided training set (penalty: 0 points if violated)
- No pre-trained models
- 1D/2D convolution for CNN allowed
- Flexibility in modifying code and modeling, provided input/output shapes are maintained
- Techniques allowed: Data augmentation, Teacher forcing strategy, Hyper-parameter tuning, Additional Decoder architecture (e.g., Transformer for Challenge task), Additional module (e.g., Attention module for Challenge task)
- No installation of additional packages

## Project Structure

- `data_utils.py`: Defines the dataset required for the dataloader
- `modeling.py`: Contains the CustomCNN, Encoder, and Decoder classes
- `main.ipynb`: Contains the training and evaluation workflow for the Normal task
- `main_challenge.ipynb`: Contains the training and evaluation workflow for the Challenge task
- `make_submission_file.ipynb`: Generates the submission file
- `report formats.zip`: Contains report format templates
- `data_final.tar.gz`: Contains the dataset (download link provided in the PDF)

## Evaluation

- **Accuracy Metric:**
  - Correct if the model predicts all characters of the word correctly
  - Example: If `Results = ['banana', 'aplle', 'orange', 'peenut']` and `Labels = ['banana', 'apple', 'orange', 'peanut']`, the accuracy is 2/4 = 0.5


## Additional Information
- Kaggle competition (https://www.kaggle.com/t/e328d6d647064f698acf07d6723ccb94) to evaluate the challenge task
