## This file contains the non-final version of the data analyses for
##  Expt 1 presented in the paper:
#   Shravan Vasishth, Katja Suckow, Richard Lewis, and Sabine Kern.
#   Short-term forgetting in sentence comprehension:
#   Crosslinguistic evidence from head-final structures.
#   Submitted to Language and Cognitive Processes, 2007.

library(lme4)

#library(Hmisc)
library(xtable)
library(MASS)

library(tidyr)
library(dplyr)
## preliminary data processing:
data <- read.csv("../../../../stimuli/V11/V11_German_adapted_tokenized.txt", sep="\t")
data$LineNumber = (1:nrow(data))-1

modelsTable = read.csv("../../../results/models_bottlenecked_german", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("../../../output/V11_E3_Adapted_german_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL)
datModel = merge(datModel, modelsTable, by=c("Model"))

data = merge(data, datModel, by=c("LineNumber"))

data = data %>% filter(!grepl("OOV", RegionLSTM))



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

summary(fm3 <- lmer(Surprisal~ g+i+gxi +(1+g|Model)+(1+g|Item), data=data3)) # this is the effect on the verb


library(ggplot2)
plot = ggplot(data3 %>% group_by(g, Model, Item) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Item)


data3 = data3 %>% mutate(Bottleneck = (!grepl("Control", Script))) %>% mutate(LogBeta = ifelse(Bottleneck, LogBeta, NA)) %>% mutate(grammatical = ifelse(g==-1, "Ungrammatical", "Grammatical"))
library(ggplot2)
plot = ggplot(data3 %>% group_by(grammatical, Model, LogBeta) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~LogBeta)
plot = ggplot(data3 %>% group_by(grammatical, Model, Memory) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Memory)
plot = ggplot(data3 %>% group_by(grammatical, Model, NumberOfDevRuns) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~NumberOfDevRuns)
plot = ggplot(data3 %>% group_by(grammatical, Model, ModelPerformance) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_point() + facet_wrap(~ModelPerformance)




## comparison 4:
data4 <- subset(data,(Region=="D4"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|Item), data=data4))


library(ggplot2)
plot = ggplot(data4 %>% group_by(g, Model, Item) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Item)




data4 = data4 %>% mutate(LogBeta.C = LogBeta-mean(LogBeta))
#summary(fm4a <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1|item), data=data4))

data4 = data4 %>% mutate(ModelPerformance.C = ModelPerformance-mean(ModelPerformance))
#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C*g+(1|Model)+(1|item), data=data4 %>% filter(!grepl("Control", Script))))
#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C*g+(1|Model)+(1|item), data=data4 %>% filter(grepl("Control", Script))))
#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C+g+(1|Model)+(1|item), data=data4 %>% filter(grepl("Control", Script))))  


#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C*g+(1|Model)+(1|item), data=data4)) # Degree of grammaticality advantage is modulated by model surprisal


data4 = data4 %>% mutate(Bottleneck = (!grepl("Control", Script))) %>% mutate(LogBeta = ifelse(Bottleneck, LogBeta, NA)) %>% mutate(grammatical = ifelse(g==-1, "Ungrammatical", "Grammatical"))
library(ggplot2)
plot = ggplot(data4 %>% group_by(grammatical, LogBeta, Model) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~LogBeta)
plot = ggplot(data4 %>% group_by(grammatical, NumberOfDevRuns, Model) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~NumberOfDevRuns)

#plot = ggplot(data4 %>% group_by(grammatical) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal)) + geom_line() #(stat="identity", position=position_dodge(0.9))


## comparison 5:
data5 <- subset(data,(Region=="N4"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|Item), data=data5))


library(ggplot2)
plot = ggplot(data5 %>% group_by(g, Model, Item) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Item)




data5 = data5 %>% mutate(LogBeta.C = LogBeta-mean(LogBeta))
#summary(fm4a <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1|item), data=data5))

data5 = data5 %>% mutate(ModelPerformance.C = ModelPerformance-mean(ModelPerformance))
#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C*g+(1|Model)+(1|item), data=data5 %>% filter(!grepl("Control", Script))))
#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C*g+(1|Model)+(1|item), data=data5 %>% filter(grepl("Control", Script))))
#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C+g+(1|Model)+(1|item), data=data5 %>% filter(grepl("Control", Script))))  


#summary(fm4a <- lmer(Surprisal~ ModelPerformance.C*g+(1|Model)+(1|item), data=data5)) # Degree of grammaticality advantage is modulated by model surprisal


data5 = data5 %>% mutate(Bottleneck = (!grepl("Control", Script))) %>% mutate(LogBeta = ifelse(Bottleneck, LogBeta, NA)) %>% mutate(grammatical = ifelse(g==-1, "Ungrammatical", "Grammatical"))
library(ggplot2)
plot = ggplot(data5 %>% group_by(grammatical, LogBeta, Model) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~LogBeta)
#plot = ggplot(data5 %>% group_by(grammatical) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal)) + geom_line() #(stat="identity", position=position_dodge(0.9))
plot = ggplot(data5 %>% group_by(grammatical, Model, NumberOfDevRuns) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~NumberOfDevRuns)




data$gram <- ifelse(data$Condition%in%c("a","b"),"gram","ungram")
data$int <- ifelse(data$Condition%in%c("a","c"),"hi","lo")



#data = data %>% mutate(positionR = ifelse(gram == "gram" | position < 10, position, position+1))
plot = ggplot(data %>% mutate(LogMemory = log(Memory+1)) %>% group_by(Model, Region, gram, LogMemory) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Region, y=Surprisal, group=paste(Model, gram, LogMemory), color=LogMemory)) + geom_line(aes(linetype=gram))
plot = ggplot(data %>% group_by(Model, Region, gram, Memory) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Region, y=Surprisal, group=paste(Model, gram))) + geom_line(aes(linetype=gram)) + facet_wrap(~Memory) 


plot = ggplot(data %>% group_by(positionR, gram) %>% summarise(Surprisal = mean(Surprisal)), aes(x=positionR, y=Surprisal, group=gram, color=gram)) + geom_line()




