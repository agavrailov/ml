tsteps <- 1 # since we are using stateful rnn tsteps can be set to 1
batch_size <- 20
epochs <- 25
lahead <- 1 # number of elements ahead that are used to make the prediction

library('keras', quietly = T)
library('caret', quietly = T)

neural.train = function(model,XY) 
{
  X <- data.matrix(XY[,-ncol(XY)])
  Y <- XY[,ncol(XY)]
  # Y <- ifelse(Y > 0,1,0)
  Model <- keras_model_sequential() 
  Model %>%
    layer_lstm(units = 5, 
               input_shape = c(tsteps,1),
               batch_size = batch_size,
               return_sequences = TRUE, 
               stateful = TRUE) %>% 
    layer_dropout(rate = 0.2) %>%
    layer_lstm(units = 10,
               return_sequences = FALSE, 
               stateful = TRUE) %>% 
    layer_dense(units = 1)
  Model %>% compile(
              loss = 'mse', 
              optimizer = 'rmsprop', 
              metrics = c('accuracy'))
  
  Model %>% fit(X, Y, 
    epochs = epochs, 
    batch_size = batch_size, 
    validation_split = 0, 
    shuffle = FALSE)
  
  Model %>% reset_states()
  Models[[model]] <<- Model
}

neural.predict = function(model,X)
{
  if(is.vector(X)) X <- t(X)
  X <- as.matrix(X)
  # Y <- Models[[model]] %>% predict(X) %>% `>`(0.5) %>% k_cast("int32")
  Y <- Models[[model]] %>% predict(X, batch_size = batch_size)
  # return(ifelse(Y > 0.5,1,0))
  return(Y)
}

neural.save = function(name)
{
  for(i in c(1:length(Models)))
    Models[[i]] <<- serialize_model(Models[[i]])
  save(Models,file=)  
}

neural.load <- function(name)
{
  load(name,.GlobalEnv)
  for(i in c(1:length(Models)))
    Models[[i]] <<- unserialize_model(Models[[i]])
}

neural.init = function()
{
  set.seed(365)
  Models <<- vector("list")
}

neural.test = function() 
{
  neural.init()
  XY <<- read.csv("D:\\My Documents\\R\\ml\\data\\training_data.csv",header = TRUE)
  # XY <<- XY[c("Open.1","High.1", "Low.1", "Label1")]
    XY <<- XY[c("Open", "Label1")]
    XY <<- head(XY, nrow(XY)-nrow(XY) %% batch_size)   #make the whole set divisible by batch size
  
  splits <- nrow(XY)*0.8

  XY.tr <<- head(XY,splits)
  XY.ts <<- tail(XY,-splits)
  neural.train(1,XY.tr)
  
  X <<- XY.ts[,-ncol(XY.ts)]
  X <- array(data = X, dim = c(length(X), 1)) #array needed for input of fit(). TODO May be there is a better structure
  Y <<- XY.ts[,ncol(XY.ts)]

  Y.ob <<- Y                    #observed values
  Y.pr <<- neural.predict(1,X)  #predicted values

  op <- par(mfrow=c(3,1))
  plot(Y.ob, xlab = 'Expected')
  plot(Y.pr, xlab = 'Predicted')
  plot(Y.ob-Y.pr, xlab = 'Diff')
  sd(Y.ob-Y.pr)

  # Y.ob <<- ifelse(Y > 0,1,0)    #observed values
  # Y.pr <<- neural.predict(1,X)  #predicted values
  # confusionMatrix(as.factor(Y.pr),as.factor(Y.ob))
}

neural.test()

