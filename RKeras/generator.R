# library(Keras)
tsteps = 5  #window size
lag = 10  #Labels number of rows ahead
batch_size = 20
epochs = 20

XY <- read.csv("D:\\My Documents\\R\\ml\\data\\training_1c.csv",header = TRUE)
# XY <- head(XY,
           # trunc( nrow(XY)/batch_size) *  batch_size)
X <- as.matrix(XY["Open.1"])
Y <-rbind(matrix(rep(mean(X),lag)),head(X,-lag))   #lagged version of training data

generator = timeseries_generator(X,Y, 
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(XY)-batch_size,
                                 sampling_rate = 1,
                                 stride = 3,
                                 shuffle = FALSE)

Model <- keras_model_sequential() 
Model %>%
  layer_lstm(units = 5, 
             input_shape = c(tsteps, ncol(X)),
             batch_size = batch_size,
             return_sequences = TRUE, 
             stateful = TRUE) %>% 
  layer_dropout(rate = 0.0) %>%
  layer_lstm(units = 5,
             return_sequences = FALSE, 
             stateful = TRUE) %>% 
  layer_dense(units = 1)
Model %>% compile(
  loss = 'mse', 
  optimizer = 'rmsprop', 
  metrics = c('accuracy'))

Model %>% fit(generator, 
              batch_size = batch_size,
              epochs = epochs,
              ) 

# TODO fix batchsize. works only with  20
# make a one step prediction out of sample
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
