library(Keras)
tsteps = 5  #window size
lag = 10  #Labels number of rows ahead
batch_size = 20
epochs = 20

XY <<- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
XY <<- XY[c("Time","Open.1","High.1", "Label1")]
X <- as.matrix(XY[2])
# X <- as.matrix(seq(1:20))   #used to test the lagging and timeseries_generator
Y <-rbind(matrix(rep(mean(X),lag)),head(X,-lag))   #lagged version of trainingg data

# X <- XY[,-ncol(XY)]
# Y <- as.matrix(XY[,ncol(XY)])

generator = timeseries_generator(X,Y, 
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(XY)-batch_size,
                                 sampling_rate = 1,
                                 stride = 3,
                                 shuffle = FALSE)
# length(generator)
# for(i in seq(1:length(generator))){
#   x = y = generator[i]
#   print(x)
# }

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

Model %>% fit(generator, 
              batch_size = 10,
              epochs = epochs)

# TODO make a one step prediction out of sample
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
