suppressMessages(library(ggplot2)) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(tm)
suppressMessages(library(wordcloud))
suppressMessages(library(plyr))
suppressMessages(library(lubridate))
suppressMessages(library(syuzhet))
suppressMessages(library(dplyr))
library(forecast)

tweets1 <- read.csv("RedditNews.csv")
time_Stamp <- ymd(tweets1$Date)
tweets1$Date <- time_Stamp


some_txt<-gsub("(RT|via)((?:\\b\\w*@\\w+)+)","",tweets1$News)
some_txt<-gsub("http[^[:blank:]]+","",some_txt)
some_txt<-gsub("@\\w+","",some_txt)
some_txt<-gsub("[[:punct:]]"," ",some_txt)
some_txt<-gsub("[^[:alnum:]]"," ",some_txt)


tweet1Sentiment <- get_nrc_sentiment(some_txt)

tweet1Sentiment$Date <- tweets1$Date

Table1 <-  tweet1Sentiment%>%
  group_by(Date) %>%
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
write.csv(Table1,"Sent_scores.csv")