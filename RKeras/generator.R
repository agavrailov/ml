library(keras)
tsteps = 3  #window size
rows_ahead = 5  #prediction Labels are n rows ahead of the current
batch_size = 200
epochs = 40
split = 0.7   #part of data used for training 
LSTM_units = 50


XY <- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
# XY<-as.data.frame(1:200)

# XY.tr training set
XY.tr <- head(XY,nrow(XY)*split)
extra_rows <- (nrow(XY.tr)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
extra_rows
if (extra_rows) XY.tr<-head(XY.tr,-extra_rows)

# XY.val validation set
XY.val <-tail(XY, -split*nrow(XY))
extra_rows <- (nrow(XY.val)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
extra_rows
if (extra_rows) XY.val<-head(XY.val,-extra_rows)

data.frame(nrow(XY), nrow(XY.tr), nrow(XY.val))

#the generator
# X <- as.matrix(XY.tr[ncol(XY.tr)])
X <- as.matrix(XY.tr["Open.1"])
Y <-rbind(matrix(rep(mean(X)),rows_ahead),head(X,-rows_ahead))  #Create lagged version of training data
generator = timeseries_generator(X,Y, 
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X)-1,
                                 sampling_rate = 1,
                                 stride = 1,
                                 shuffle = FALSE)

#the generator for validation
X.val <- as.matrix(XY.val[ncol(XY.val)])
X.val <- as.matrix(XY.val["Open.1"])
Y.val <-rbind(matrix(rep(mean(X.val)),rows_ahead),head(X.val,-rows_ahead))  #Create lagged version of training data
generator.val = timeseries_generator(X.val,
                                 Y.val,
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X.val)-1,
                                 sampling_rate = 1,
                                 stride = 1,
                                 shuffle = FALSE)
# for(i in seq(1:length(generator.val))){
#   x = y = generator[i]
#   print(x)
# }

Model <- keras_model_sequential() 

Model %>%
  layer_lstm(units = LSTM_units, 
             input_shape = c(tsteps,ncol(X)),
             batch_size = batch_size,
             return_sequences = TRUE, 
             stateful = TRUE) %>% 
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = LSTM_units,
             return_sequences = FALSE, 
             stateful = TRUE) %>% 
  layer_dense(units = 1)

Model %>% compile(
  loss = 'mse', 
  optimizer = 'rmsprop', 
  metrics = c('accuracy'),
  )

Model %>% fit(generator, 
              batch_size = batch_size,
              epochs = epochs,
              validation_data = generator.val
              ) 

# TODO 
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
