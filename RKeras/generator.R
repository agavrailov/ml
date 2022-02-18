library(keras)
tsteps = 5  #window size
rows_ahead = 5  #prediction Labels are n rows ahead of the current
batch_size = 64
epochs = 40
split = 0.7   #part of data used for training 
LSTM_units = 30

XY <- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
XY <- XY[c("Open","High","Low","Close","Label1")]  #add as many columns as we need

# XY.tr training set
XY.tr <- head(XY,nrow(XY)*split)
extra_rows <- (nrow(XY.tr)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
if (extra_rows) XY.tr<-head(XY.tr,-extra_rows)

# XY.val validation set
XY.val <-tail(XY, -split*nrow(XY))
extra_rows <- (nrow(XY.val)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
if (extra_rows) XY.val<-head(XY.val,-extra_rows)

#the generator
X <- as.matrix(XY.tr[,-ncol(XY.tr)])  #all, but last column
Y <- as.matrix(XY.tr[, ncol(XY.tr)])  #last column
Y <- rbind(tail(Y,-rows_ahead),as.matrix(rep(mean(tail(Y,-rows_ahead)),rows_ahead))) #Create lagged version of last column
generator = timeseries_generator(X,Y, 
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X)-1,
                                 sampling_rate = 1,
                                 stride = 1,
                                 shuffle = FALSE)

#the generator for validation
X.val <- as.matrix(XY.val[,-ncol(XY.val)])  #all, but last column
Y.val <- as.matrix(XY.val[, ncol(XY.val)])  #last column
Y.val <- rbind(tail(Y.val,-rows_ahead),as.matrix(rep(mean(tail(Y.val,-rows_ahead)),rows_ahead))) #Create lagged version of last column

generator.val = timeseries_generator(X.val,
                                 Y.val,
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X.val)-1,
                                 sampling_rate = 1,
                                 stride = 1,
                                 shuffle = FALSE)

Model <- keras_model_sequential() 

Model %>%
  layer_lstm(units = LSTM_units, 
             input_shape = c(tsteps,ncol(X)),
             batch_size = batch_size,
             return_sequences = TRUE, 
             stateful = TRUE,
             activation = 'tanh') %>% 
  layer_dropout(rate = 0.1) %>%
  
  layer_lstm(units = LSTM_units,
             return_sequences = FALSE, 
             stateful = TRUE,
             activation = 'tanh') %>% 
  layer_dense(units = 1)

Model %>% compile(
  loss = 'mse', 
  optimizer = optimizer_rmsprop(
    learning_rate = 0.001), 
  metrics = c('accuracy'),
  )

Model %>% fit(generator, 
              batch_size = batch_size,
              epochs = epochs,
              validation_data = generator.val
              ) 

# TODO 
# reshape input to be [samples, time steps, features]
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))


# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
