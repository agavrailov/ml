library(keras)
neural.generate = function(XY) {
X <- as.matrix(XY[,-ncol(XY)])  #all, but last column
Y <- as.matrix(XY[, ncol(XY)])  #last column
Y <- rbind(tail(Y,-rows_ahead),as.matrix(rep(mean(tail(Y,-rows_ahead)),rows_ahead))) #Create lagged version of last column
generator = timeseries_generator(X,Y, 
                                 length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X)-1,
                                 sampling_rate = 1,
                                 stride = 1,
                                 shuffle = FALSE)
return(generator)
}
neural.train = function(model, generator,generator.val) {
  Model <- keras_model_sequential() 
  Model %>%
    layer_lstm(units = LSTM_units, 
               input_shape = c(tsteps,ncol(X)),
               batch_size = batch_size,
               return_sequences = TRUE, 
               stateful = TRUE,
               activation = 'tanh',
               ) %>% 
    layer_dense(units = 1)
  Model %>% compile(
                    loss = 'mse', 
                    optimizer = optimizer_rmsprop(
                                                  learning_rate = 0.001,
                                                  ),
                    metrics = c('accuracy'))
  Model %>% fit(generator, 
              batch_size = batch_size,
              epochs = epochs,
              validation_data = generator.val) 
  
  Models[[model]] <<- Model
}
neural.predict = function(model,X) {
  if(is.vector(X)) X <- t(X)
  X <- as.matrix(X)
  X <- array_reshape(X,c(tsteps,ncol(X))) #The LSTM expects data input to have the shape [samples, timesteps, features]
  Y <- Model %>% predict(X, batch_size = batch_size)
  return(Y)
}
neural.save = function(name) {
  for(i in c(1:length(Models)))
    Models[[i]] <<- serialize_model(Models[[i]])
  save(Models,file=name) 
}

neural.load <- function(name) {
  load(name,.GlobalEnv)
  for(i in c(1:length(Models)))
    Models[[i]] <<- unserialize_model(Models[[i]])
}

neural.init = function() {
  set.seed(365)
  input_cols <- c("Open.1","High.1","Low.1","Close.1","Label1") #add as many columns as we need
  XY <- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)[input_cols]
  
  # XY.tr training set
  XY.tr <- head(XY,nrow(XY)*tr_split)
  extra_rows <- (nrow(XY.tr)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
  if (extra_rows) XY.tr<-head(XY.tr,-extra_rows)
  
  # XY.val validation set
  XY.val <-tail(XY, -tr_split*nrow(XY))
  extra_rows <- (nrow(XY.val)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
  if (extra_rows) XY.val<-head(XY.val,-extra_rows)
  
  generator <- neural.generate(XY.tr)
  generator.val <- neural.generate(XY.val)
  
  neural.train(1,generator, generator.val)
  neural.predict(1,XY.val)
 
}
Models <<- vector("list")
tsteps <- 4  #window size a.k.a. time steps
rows_ahead <- 5  #prediction Labels are n rows ahead of the current
batch_size <- 500
epochs <- 5
tr_split <- 0.7   #part of data used for training 
LSTM_units <- 5

neural.init()


# TODO 
# reshape input to be [samples, time steps, features]
# X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))


# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
