library('keras')
setwd("C:\\Users\\Anton\\anaconda3\\envs\\rstudio\\src\\data") 
inputfile <- "training_data_4variables.csv"

#load input timeseries file
df <- read.csv(inputfile, header = TRUE)

neural.train = function(model,XY)
{
  X <- data.matrix(XY[,-ncol(XY)])
  Y <- XY[,ncol(XY)]
  # Y <- ifelse(Y > 0,1,0)
  Model <- keras_model_sequential()
  Model %>%
    
    layer_dense(units=1,activation='relu',input_shape = c(ncol(X), batch_size = 32)) %>%
    layer_dropout(rate = 0.2) %>%
    layer_rnn()
    layer_dense(units = 1, activation = 'sigmoid')
  
  Model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'))
  
  Model %>% fit(X, Y,
                epochs = 20, batch_size = 20,
                validation_split = 0.3, shuffle = FALSE)
  
  Models[[model]] <<- Model
}

neural.predict = function(model,X)
{
  if(is.vector(X)) X <- t(X)
  X <- as.matrix(X)
  # Y <- Models[[model]] %>% predict(X) %>% `>`(0.5) %>% k_cast("int32")
  Y <- Models[[model]] %>% predict(X)
  
  return(Y)
  
}

neural.save = function(name)
{
  for(i in c(1:length(Models)))
    Models[[i]] <<- serialize_model(Models[[i]])
  save(Models,file=name) 
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

#Prepare the table input table and fields
df_t<-df[c("Open.1","High.1", "Low.1", "Label1")]
df_p<-df[c("Label1")]

neural.train(1, df_t)

a <- neural.predict(1, df_p)
summary (a)