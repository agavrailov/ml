library(keras)
tsteps = 10  #window size
rows_ahead = 1  #prediction Labels are n rows ahead of the current
batch_size = 32
epochs = 5
split = 0.8   #part of data used for training 

XY <- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
extra_rows <- (nrow(XY)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
extra_rows
if (extra_rows) 
  XY<-head(XY,-extra_rows)

# Split training validation and test sets
XY.tr <- head(XY,nrow(XY)*split)
XY.val <- tail(XY.tr, -split*nrow(XY.tr))
XY.tr <- head(XY.tr, split*nrow(XY.tr))
XY.ts <- tail(XY,-split*nrow(XY))
data.frame(nrow(XY), nrow(XY.tr), nrow(XY.val), nrow(XY.ts))

#Create lagged version of training data
# X <- as.matrix(XY.tr[ncol(XY.tr)])
X <- as.matrix(XY[ncol(XY)])
# Y <-rbind(matrix(rep(mean(X)),rows_ahead),head(X,-rows_ahead))  
# rownames(Y) <- NULL

#the generator
generator = timeseries_generator(X,X, 
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
              epochs = epochs
              ) 

# TODO fix batchsize. works only with  20
# make a one step prediction out of sample
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# print(yhat)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
