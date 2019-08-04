###################################################
### chunk number 2: 
###################################################

library(reshape)
library(lme4)
#library(Hmisc)
library(tidyr)
library(dplyr)

models = c(
710899603  , 
553555034   ,
657350374   ,
887261094   ,
459934402   
)

dataModels = data.frame()

for(model in models) {
  data2 = read.csv(paste("../../../output/V11_E3_Adapted_german_", model, sep=""), sep="\t")  %>% mutate(Model = model)
  dataModels = rbind(dataModels, data2)
}

data <- read.csv("../../../../stimuli/V11/V11_German_adapted_tokenized.txt", sep="\t")
data$LineNumber = (1:nrow(data))-1

data = merge(data, dataModels, by=c("LineNumber")) %>% filter(!grepl("OOV", RegionLSTM))

mean(as.character(data$RegionLSTM) == as.character(data$Word))



data$g <- ifelse(data$Condition%in%c("a","b"),1,-1)
data$i <- ifelse(data$Condition%in%c("a","c"),1,-1)
data$gxi <- ifelse(data$Condition%in%c("a","d"),1,-1)



## comparison 0:
data0 <- subset(data,Region=="DN3")

summary(fm0 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|Item),
                   data=data0))

## comparison 1:
data1 <- subset(data,Region=="V3")

summary(fm1 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|Item),
                   data=data1))


## comparison 2:

data2 <- subset(data,(Region=="V2" & Condition%in%c("a","b")) |
                                        (Region=="V1" & Condition%in%c("c","d")))

summary(fm2 <- lmer(Surprisal~ g+i+gxi+
                    #center(wl)+
                    (1|Model),#+(1|Item),
                    data=data2))



## comparison 3:
data3 <- subset(data,(Region=="V1"))

summary(fm3 <- lmer(Surprisal~ g+i+gxi +(1+g|Model)+(1+g|Item), data=data3)) # this is the effect on the verb (nothing found here)


library(ggplot2)
plot = ggplot(data3 %>% group_by(g, Model, Item) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Item)


## comparison 4:
data4 <- subset(data,(Region=="D4"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|Item), data=data4)) # the opposite of structural forgetting (expected)

library(ggplot2)
plot = ggplot(data4 %>% group_by(g, Model, Item) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Item)




## comparison 4:
data5 <- subset(data,(Region=="N4"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|Item), data=data5))




plot = ggplot(data %>% group_by(Model, Region, g) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Region, y=Surprisal, group=paste(Model, g))) + geom_line(aes(linetype=as.factor(g)))




