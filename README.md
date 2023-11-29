# Sequence-Models
This repository contains code for a simple time series prediction using linear regression. The code is implemented in PyTorch and utilizes the d2l library for utility functions.
##Dependencies

Make sure you have the required dependencies installed:

bash

pip install d2l

##Dataset

The dataset consists of a synthetic time series generated with a sine function and some random noise. The goal is to predict future values in the time series.
Model

The predictive model is a simple linear regression model implemented with PyTorch. The training process and evaluation are facilitated by the d2l library.
Usage

    Install dependencies: pip install d2l
    Run the provided code in a Python environment.
    View the generated plots showing one-step and multi-step predictions.

Code Structure

    linear_regression.py: Defines the linear regression model.
    data_module.py: Implements a data module for handling the time series data.
    train.py: Script for training the model.
    predict.py: Script for generating predictions.

#Results

The repository includes visualizations of one-step and multi-step predictions, showcasing the model's performance.
Acknowledgments

The code is based on the deep learning framework provided by the d2l library, which offers a set of convenient tools for educational purposes.

Feel free to explore, modify, and contribute to improve the model or extend its functionality.
