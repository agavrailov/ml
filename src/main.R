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
                                learning_rate = learning_rate),
                    metrics = c('accuracy'))
  Model %>% fit(generator, 
              epochs = epochs,
              validation_data = generator.val)
              # samples_per_epoch = floor(length(XY)/batch_size) 
  
  Models[[model]] <<- Model
}
neural.predict  <- function(model,X) {
  Y <- Models[[model]] %>% predict(X)
  Y.denorm = t(apply(Y, 1, function(r)r*attr(XY_norm,'scaled:scale') + attr(XY_norm, 'scaled:center')))
  return(Y.denorm)
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
neural.plot     <- function(){
  ###Plotting results ------------
  cat('Plotting Results\n')
  op <- par(mfrow=c(3,1))
  plot(XY.tr[,1], xlab = '')
  title("Expected")
  
  plot(Y.pred[,1], xlab = '')   #dim(Y_pred) {20500,5}, използваме само първия елемент. TODO да се помисли за apply(Y_pread(1:5),1, mean)
  title("Predicted")
  
  XY.tr<-head(XY.tr, nrow(Y.pred))  #XY.tr is 4 rows longer than the result
  delta <- XY.tr[,1]- Y.pred[,1]
  plot(delta, xlab = '')
  title("Difference")
}

### Constants  ----------------------------------
Models <<- vector("list")
tsteps <- 3  #window size a.k.a. time steps
rows_ahead <- 60  #prediction Labels are n rows ahead of the current
tr_split <- 0.7   #part of data used for training 
nfeatures <- 1    #how many columns we will use
set.seed(365)
setwd("D:\\My Documents\\R\\ml\\") 
inputfile <-"data\\training_data"

batch_size <- 500
epochs <- 20
learning_rate <-0.001
LSTM_units <- 50  #number of neurons in a LSTM layer


### Data pre-processing  ----------------------------------
load(inputfile)  # XY <- read.csv(file = paste(outputfile,".csv", sep = ), header = TRUE)
XY <- XY[c("Open","High","Low","Close")]  #add as many columns as we need c("Open.1","High.1","Low.1","Close.1","Label1")
XY<-XY[1:nfeatures]   #use only 1-4 columns. It is a data frame

XY.norm <- scale(XY)    #it is a matrix already

XY.tr <- neural.datasets(XY, tr_split)
XY.tr.norm <- neural.datasets(XY_norm, tr_split)
XY.val.norm<- neural.datasets(XY_norm, (1-tr_split))


generator <- neural.generate(XY.tr.norm)
generator.val <- neural.generate(XY.val.norm)

### Training and predicting  ----------------------------------
neural.train(1,generator, generator.val, ncol(XY.tr.norm))
Y.pred <-neural.predict(1,neural.generate(XY.tr.norm))
neural.predict(1, as.data.frame(c(1,1,1,1)))
neural.plot()

par(op)
print(paste("Standard deviation",sd(delta)))

### Saving models
neural.save("models\\MyModels")


# TODO 
# https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/