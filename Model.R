
FIN_Data <- read.csv("fib.csv")
time_Stamp <- dmy_hm(FIN_Data$Date.Time)
FIN_Data$Date <- date(time_Stamp)
FIN_Data$hour <- hour(time_Stamp)
Table_Fin <-  FIN_Data %>%
  group_by(Date,hour) %>%
  summarise(Open_price = mean(Open),
            Close_price = mean(Close))

Table1[,3:12] <- as.data.frame(lapply(Table1[,3:12],function(x){ma(x,5)}))
Table1 <- Table1[-c(1,2,399,400),]
library(dplyr)
Fin_df <- inner_join(Table1, Table_Fin)


set.seed(1200)
train = 1:60
#Train <- Fin_df[ 1:60 ,c(3:13)]
#Test <- Fin_df[61:83, c(3:13)]
Train <- Fin_df[train,c(3:13)]
Test  <- Fin_df[-train,c(3:13)]

################################## RandomForest ########################################

library(randomForest)
response <- as.vector( Test[,11])
model <- randomForest(Open_price ~ ., data = Train,ntree = 1000, mtry = 5)
pred <- predict(model, newdata = Test)
plot(pred,type = "l",ylim = c(115,120),col = "red",xlab = "Day No",ylab = "Stock price($)")
lines(response,col = "green", lty = 2)
legend("bottomright",c("Predicted Stock Price","Actual Stock Price"),col = c("red","green"),lty = 1:2)


######################################### Neural Network #############################

library(neuralnet)
#nn <- neuralnet(Open_price ~ .,hidden=c(5,3),linear.output=T,data = Train)
n <- names(Train)
f <- as.formula(paste("Open_price ~", paste(n[!n %in% "Open_price"], collapse = " + ")))
nn <- neuralnet(f,data=Train,hidden=c(5,3),linear.output=T)
pr.nn <- compute(nn,Test[,1:10])
pr.nn_ <- pr.nn$net.result*(max(dv)-min(data$medv))+min(data$medv)
plot(pr.nn,type = "l",ylim = c(115,120),col = "red",xlab = "Day No",ylab = "Stock price($)")
lines(response,col = "green", lty = 2)
legend("bottomright",c("Predicted Stock Price","Actual Stock Price"),col = c("red","green"),lty = 1:2)