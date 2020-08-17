suppressMessages(library(ggplot2)) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(tm)
suppressMessages(library(wordcloud))
suppressMessages(library(plyr))
suppressMessages(library(lubridate))
suppressMessages(library(syuzhet))
suppressMessages(library(dplyr))
library(forecast)

setwd("C:/Users/Gaurav kumar/Desktop/Sentiment Analysis/demonetization-in-india-twitter-data")
tweets1 <- read.csv("tweet.csv")
time_Stamp <- ymd_hms(tweets1$timestamp)
tweets1$Date <- date(time_Stamp)
tweets1$hour <- hour(time_Stamp)


some_txt<-gsub("(RT|via)((?:\\b\\w*@\\w+)+)","",tweets1$text)
some_txt<-gsub("http[^[:blank:]]+","",some_txt)
some_txt<-gsub("@\\w+","",some_txt)
some_txt<-gsub("[[:punct:]]"," ",some_txt)
some_txt<-gsub("[^[:alnum:]]"," ",some_txt)


tweet1Sentiment <- get_nrc_sentiment(some_txt)
tweets1$positive <- tweet1Sentiment$positive
tweets1$negative <- tweet1Sentiment$negative

tweet1Sentiment$Date <- tweets1$Date
tweet1Sentiment$hour <- tweets1$hour

Table1 <-  tweet1Sentiment%>%
  group_by(Date,hour) %>%
  summarise(PosCount = sum(positive),
            NegCount = sum(negative),
            TrustCount = sum(trust),
            AngerCount = sum(anger),
            AnticipationCount = sum(anticipation),
            DisgustCount = sum(disgust),
            FearCount = sum(fear),
            JoyCount = sum(joy),
            SadnessCount = sum(sadness),
            SurpriseCount = sum(surprise))

Table1$PosProp <- Table1$PosCount / (Table1$PosCount + Table1$NegCount)
Table1$NegProp <- 1 - Table1$PosProp



barplot(
  sort(colSums(prop.table(tweet1Sentiment[, 1:10]))), 
  horiz = TRUE, 
  cex.names = 0.7, 
  las = 1, 
  main = "Emotions in Tweets text", xlab="Percentage"
)