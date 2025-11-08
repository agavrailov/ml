library(xts)
setwd("D:\\My Documents\\R\\ml\\data\\") 
inputfile <- "xagusd_hourly.csv"

#load input timeseries file
XY <- read.csv(inputfile, header = TRUE)
XY<- XY[1:5]  #in case of minute file, cut the extra columns
colnames(XY)<-c("DateTime", "Open","High","Low","Close")

# normalize OHLC values for much better training accuracy
# Z score implementation df$Open = ((df$Open - mean(df$Open)) / sd(df$Open))
# scale(x, center = FALSE, scale = apply(x, 2, sd, na.rm = TRUE))
XY[6:9]<- as.data.frame(scale(XY[2:5]))

# Label1 <- tail(df$Open,-)
# df<-head(df,-p3) 
# df["Label3"]<- price_p3

outputfile <-"training_data"
write.csv(XY, file = paste(outputfile,".csv", sep =""), row.names = FALSE)
save(XY, file = outputfile)
