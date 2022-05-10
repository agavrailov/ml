library(keras)
neural.generate <- function(XY) {
  X <- as.matrix(XY)
  Y <- as.matrix(XY[, 1])  #first  column is used for labels
  ins_value_for_rows_ahead<-rep(mean(head(Y,rows_ahead)),rows_ahead) #what to insert in beginning
  Y <- rbind(as.matrix(ins_value_for_rows_ahead),tail(Y,-rows_ahead))  #Create lagged version of first column
  rownames(Y)<-NULL
  
  dataset = timeseries_dataset_from_array(X,Y, 
                                sequence_length = tsteps, 
                                 batch_size = batch_size, 
                                 start_index = 1, 
                                 end_index = nrow(X)-1,
                                 sampling_rate = 1,
                                 sequence_stride = 1,
                                 shuffle = FALSE)
return(dataset)
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
              epochs = epochs,
              validation_data = generator.val)
              # samples_per_epoch = floor(length(XY)/batch_size) 
  
  Models[[model]] <<- Model
}
neural.predict  <- function(model,X) {
  # if (round(sd(X))!=1) Y <- scale(X)    #if not scaled, scale to predict scaled, not real number
  Y <- Models[[model]] %>% predict(X)
  Y.orig = t(apply(Y, 1, function(r)r*attr(XY,'scaled:scale') + attr(XY, 'scaled:center')))
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
LSTM_units <- 50

setwd("D:\\My Documents\\R\\ml\\") 
inputfile <-"data\\training_data"
load(inputfile)  # XY <- read.csv(file = paste(outputfile,".csv", sep = ), header = TRUE)
XY <- XY[c("Open","High","Low","Close")]  #add as many columns as we need c("Open.1","High.1","Low.1","Close.1","Label1")

XY <- scale(XY)
attr(XY,'scaled:scale')
attr(XY, 'scaled:center')

set.seed(365)

XY.tr <- neural.datasets(XY, tr_split)
XY.val<- neural.datasets(XY, (1-tr_split))

generator <- neural.generate(XY.tr)
generator.val <- neural.generate(XY.val)

neural.train(1,generator, generator.val, ncol(XY))
Y_pred <-neural.predict(1,neural.generate(XY.tr))

###Plotting results ------------
cat('Plotting Results\n')
op <- par(mfrow=c(3,1))
plot(XY.tr[,1], xlab = '')
title("Expected")
plot(Y_pred[,1,1], xlab = '')
title("Predicted")
plot(XY.tr[,1]- Y_pred[,1,1], xlab = '')
title("Difference")

par(op)


#TODO ----
# neural.save("Models\\Models")

# TODO 
# x_input = array([9, 10]).reshape((1, n_input, n_features))
# yhat = model.predict(x_input, verbose=0)
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/

# x.orig = t(apply(xs, 1, function(r)r*attr(xs,'scaled:scale') + attr(xs, 'scaled:center')))
