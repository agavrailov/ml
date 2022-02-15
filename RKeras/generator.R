library(keras)
tsteps = 4  #window size
rows_ahead = 1  #prediction Labels are n rows ahead of the current
batch_size = 32
epochs = 20
split = 0.7   #part of data used for training 

XY <- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
# XY<-as.data.frame(1:200)

## Split training validation and test sets
# XY.tr
XY.tr <- head(XY,nrow(XY)*split)
extra_rows <- (nrow(XY.tr)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
extra_rows
if (extra_rows) XY.tr<-head(XY.tr,-extra_rows)

# XY.val
XY.val <-tail(XY, -split*nrow(XY))
data.frame(nrow(XY), nrow(XY.tr), nrow(XY.val))
X <- as.matrix(XY.tr[ncol(XY.tr)])

#Create lagged version of training data
Y <-rbind(matrix(rep(mean(X)),rows_ahead),head(X,-rows_ahead))  
# rownames(Y) <- NULL

#the generator
generator = timeseries_generator(X,Y, 
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X)-1,
                                 sampling_rate = 1,
                                 stride = 1,
                                 shuffle = FALSE)

# for(i in seq(1:length(generator))){
#   x = y = generator[i]
#   print(x)
# }

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
