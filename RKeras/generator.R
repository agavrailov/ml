library(keras)
neural.generate = function(XY) {
  # X <- as.matrix(XY[,-ncol(XY)])  #remove last column (in cases when last column in data is Labels)
  X <- as.matrix(XY)
  Y <- as.matrix(XY[, 1])  #first  column is used for labels
  ins_value_for_rows_ahead<-rep(mean(head(Y,rows_ahead)),rows_ahead) #what to insert in beginning
  Y <- rbind(as.matrix(ins_value_for_rows_ahead),tail(Y,-rows_ahead))  #Create lagged version of first column
  rownames(Y)<-NULL
  
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
neural.train    <- function(model, generator,generator.val,n_col) {
  Model <- keras_model_sequential() 
  Model %>%
    layer_lstm(units = LSTM_units, 
               input_shape = c(tsteps,n_col),
               batch_size = batch_size,
               return_sequences = TRUE, 
               stateful = TRUE,
               activation = 'tanh',
               ) %>% 
    layer_dense(units = 1)
  Model %>% compile(
                    loss = 'mse', 
                    optimizer = optimizer_rmsprop(
                                learning_rate = 0.001),
                    metrics = c('accuracy'))
  Model %>% fit(generator, 
              batch_size = batch_size,
              epochs = epochs,
              validation_data = generator.val) 
  
  Models[[model]] <<- Model
}
neural.predict  <- function(model,X) {
  # if(is.vector(X)) X <- t(X)
  # X <- as.matrix(X)
  # X <- array_reshape(X,c(1,tsteps,ncol(X))) #The LSTM expects data input to have the shape [samples, timesteps, features]
  Y <- Models[[model]] %>% predict(X)
  return(Y)
}
neural.save     <- function(name) {
  for(i in c(1:length(Models)))
    Models[[i]] <<- serialize_model(Models[[i]])
  save(Models,file=name) 
}
neural.load     <- function(name) {
  load(name,.GlobalEnv)
  for(i in c(1:length(Models)))
    Models[[i]] <<- unserialize_model(Models[[i]])
}
neural.datasets <- function(XY,data_split) {
  my_set <- head(XY,nrow(XY)*data_split)
  extra_rows <- (nrow(my_set)-tsteps-1) %% (batch_size)  #LSTM cells need input to be divisible by batch_size AFTER split
  if (extra_rows) my_set<-head(my_set,-extra_rows)
  return(my_set)
}

### Init ----------------------------------
Models <<- vector("list")
tsteps <- 3  #window size a.k.a. time steps
rows_ahead <- 5*60  #prediction Labels are n rows ahead of the current
batch_size <- 500
epochs <- 20
tr_split <- 0.7   #part of data used for training 
LSTM_units <- 5

setwd("D:\\My Documents\\R\\ml\\") 
outputfile <-"data\\training_data_minute"
load(outputfile)  # XY <- read.csv(file = paste(outputfile,".csv", sep = ), header = TRUE)
XY <- XY[c("Open.1","High.1","Low.1","Close.1")]  #add as many columns as we need c("Open.1","High.1","Low.1","Close.1","Label1")
set.seed(365)

XY.tr <- neural.datasets(XY, tr_split)
XY.val<- neural.datasets(XY, (1-tr_split))

generator <- neural.generate(XY.tr)
generator.val <- neural.generate(XY.val)

neural.train(1,generator, generator.val, ncol(XY))
Y_pred <-neural.predict(1,generator.val)

# neural.save("Models\\Models")

# TODO 
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
