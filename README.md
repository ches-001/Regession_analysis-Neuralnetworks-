# Regession_analysis-Neuralnetworks
This is a demo of a regression model with Neural Networks using pytorch AI framework.
It's used to predict closing value of stock given the High, Low, and Open values of stock.
The model was trained on four years of the BTC- USD stock data.
The neural network comprises of four fully connected layers with a rectify linear(ReLU) activation function inbetween each layer.
the Weights of the network were uniformly initialized to be within range of (-0.08, +0.08) and the bias were initalized to zeros(0)
the loss function used to compute the error between the output of the network and the target labels is the MSELosss (Mean Square Error Loss)
the wieights of the network are being updated by the Adam Optimizer with a learning rate of 0.001
the network was trained for 100 epochs and reached a minimum loss of 1.009.
