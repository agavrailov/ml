library(Keras)
XY <<- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
XY <<- XY[c("Time","Open.1","High.1", "Label1")]
X <- XY[2]
X <- XY[,-ncol(XY)]
Y <- as.matrix(XY[,ncol(XY)])
tsteps = 10

generator = timeseries_generator(X,X, 
                                 length = tsteps, 
                                 batch_size = 100, 
                                 start_index = 1, 
                                 end_index = nrow(XY),
                                 shuffle = FALSE)

Model <- keras_model_sequential() 
Model %>%
  layer_lstm(units = 50, 
             input_shape = c(tsteps, ncol(X)),
             batch_size = batch_size,
             return_sequences = TRUE, 
             stateful = TRUE) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 50,
             return_sequences = FALSE, 
             stateful = TRUE) %>% 
  layer_dense(units = 1)
Model %>% compile(
  loss = 'mse', 
  optimizer = 'rmsprop', 
  metrics = c('accuracy'))


for(i in seq(1:length(generator))){
  x = generator[i]
  print(c(x, y))
}

Model %>% fit(generator,
              epochs = epochs)

# make a one step prediction out of sample
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
