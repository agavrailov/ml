# Machine learning with Tensorflow/Keras/R

This project uses Deep learning technics of Tensorflow/Keras libraries for timeseries prediction.
Prediction of Silver SPOT price against USD (XAGUSD)
Used LSTM (long short term memory)

It consists of 
 - Timeseries generator
 - train
 - fit
 - evaluate
 - prediction functions.

After hundreds of experiements with neural network topology, we managed to simplify it to jsut ONE internal layer of 100 items.

Internal layers - 1
Neurons - 100
Type of cells - LSTM
Data points 69500
Prices timeframe - hourly
Data span - 2yrs

Loss 0.002
Val_loss 0.2

A key role in the project plays:
1) data preparations
  2) getting quality data
  3) converting minute to hourly timeframe
  4) clearing outliers
3) data subsetting for staisfying LSTM requirements (traingin/validation length divisable by window size and NN batch size)

R language is used for interface

Small part of the project includes also Uber's Ludwig wrapper for Tensorflow wich gave excellent results, but demonstrates no programming skills.

The project is work in progress.
