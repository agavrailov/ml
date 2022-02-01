//1. Създаваме CSV файл от Zorro history2CSV script за всяка година.
//2. Обединяваме всички CSV в един CSV "xagusd_minute.csv"
//3. Зареждаме го в скрипта
//4. Вадим "xagusd_hourly.csv" който ще се използва като тренировъчни данни в Ludwig

library(xts)

setwd("C:\\Users\\Anton\\anaconda3\\envs\\rstudio\\src\\data") 
inputfile <- "xagusd_minute.csv"

#load input timeseries file.
# Needs zoo to convert dates properly. Regular CSV throw an error in conversion
df <- read.csv.zoo(inputfile,
                   format = "%Y-%m-%dT%H:%M", tz="UTC")

# converting to hourly OHLC data
df_hourly<-to.period(df, period = "hours")

# Shift is needed to make times round.
# shift.time(index(df_hourly), n=60)

#  converting to save csv from a data.frame
df_hourly<-as.data.frame(df_hourly)
df_hourly<- cbind(row.names(df_hourly),df_hourly)


colnames(df_hourly)<-c("Time", "Open", "High", "Low", "Close")
write.csv(df_hourly, file = "xagusd_hourly.csv", col.names = TRUE, row.names = FALSE)
