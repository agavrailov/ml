library(xts)
setwd("D:\\My Documents\\R\\ml\\data\\") 
inputfile <- "xagusd_minute.csv"

#load input timeseries file
df <- read.csv(inputfile, header = TRUE)
# df<- df[1:5]
# colnames(df)<-c("DateTime", "Open","High","Low","Close")

# normalize OHLC values for much better training accuracy
# Z score implementation df$Open = ((df$Open - mean(df$Open)) / sd(df$Open))
# scale(x, center = FALSE, scale = apply(x, 2, sd, na.rm = TRUE))
df[6:9]<- as.data.frame(scale(df[2:5]))

# Label1 <- tail(df$Open,-)
# df<-head(df,-p3) 
# df["Label3"]<- price_p3

outputfile <-"training_data_minute"
write.csv(df, file = paste(outputfile,".csv", sep = ), row.names = FALSE)
save(df, file = outputfile)
