library(keras)
#load input timeseries file
setwd("C:\\Users\\Anton\\anaconda3\\envs\\rstudio\\src\\data") 
inputfile <- "training_data.csv"
cat('Loading Data...\n')
df <- read.csv(inputfile, header = TRUE)

# since we are using stateful rnn tsteps can be set to 1
tsteps <- 1
batch_size <- 25
epochs <- 25
lahead <- 100

#Prepare the table input table and fields
df<-head(df,1000*batch_size)
train_input<- array(data=df$Open.1,dim = c(nrow(df), 1))
cat('Input shape:', dim(train_input), '\n')
# df_t<-df[c("Open.1","High.1", "Low.1", "Label1")]

expected_output<-array(df$Label1,dim = c(nrow(df),1))
cat('Output shape:', dim(expected_output), '\n')

cat('Creating model:\n')
model <- keras_model_sequential()
model %>%
  layer_lstm(units = 10, 
             input_shape = c(tsteps,1),
             batch_size = batch_size,
             return_sequences = TRUE, 
             stateful = TRUE) %>% 
  layer_lstm(units = 50,
             return_sequences = FALSE, 
             stateful = TRUE) %>% 
  layer_dense(units = 1)
model %>% compile(loss = 'mse', optimizer = 'rmsprop')

cat('Training\n')
for (i in 1:epochs) {
  model %>% fit(train_input, expected_output, batch_size = batch_size,
                epochs = 1, verbose = 1, shuffle = FALSE, validation_split=0.2)
  
  model %>% reset_states()
}

cat('Predicting\n')
predicted_output <- model %>% predict(train_input, batch_size = batch_size) %>% tail(-lahead)
expected_output <-tail(expected_output,-lahead)
cat('Plotting Results\n')
op <- par(mfrow=c(2,1))
plot(expected_output, xlab = '', type = "l")
title("Expected")
plot(predicted_output, xlab = '', type = "l", col="RED" )
title("Predicted")
par(op)
