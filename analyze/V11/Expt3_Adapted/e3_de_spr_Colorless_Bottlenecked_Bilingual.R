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
dataD <- read.csv("../../../../stimuli/V11/colorlessGreen_German.txt", sep="\t")
dataD$LineNumber = (1:nrow(dataD))-1
dataD$Round = NULL
dataD$Language = 'german'
dataD$position = NULL

dataE <- read.csv("../../../../stimuli/V11/colorlessGreen_English.txt", sep="\t")
dataE$LineNumber = (1:nrow(dataE))-1
dataE$subject = NULL
dataE$rt = NULL
dataE$Language = "english"

data = rbind(dataD, dataE)

modelsTable = read.csv("../../../results/models_bottlenecked_bilingual", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(language in c("english", "german")) {
  if(language == "english") {
       exp = "E1_ColorlessGreen"
  } else {
       exp = "E3_ColorlessGreen"
  }
  for(model in models) {
     datModel2 = tryCatch(read.csv(paste("../../../output/V11_",exp,"_", language, "_", model, sep=""), sep="\t") %>% mutate(Model = model, Language=language), error=function(q) 1)
     if(datModel2 != 1) {
       cat(model,"\n")
       datModel = rbind(datModel, datModel2)
     }
  }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL)
datModel = merge(datModel, modelsTable, by=c("Model"))

data = merge(data, datModel, by=c("LineNumber", "Language"))

data = data %>% filter(!grepl("OOV", RegionLSTM))



mean(as.character(data$RegionLSTM) == as.character(data$Word), na.rm=TRUE)

######################################################

dataD = data %>% filter(Language == "german")

dataD$g <- ifelse(dataD$Condition%in%c("g"),1,-1)

######################################################

dataE = data %>% filter(Language == "english")


dataE$g <- ifelse(dataE$Condition%in%c("g"),1,-1)


library(reshape)

## computations cached:
d.rs <- dataE




d.rs.V1 <- subset(dataE,(Region=="V1"))


## V1:


## comparison 3:
data3D <- subset(dataD,(Region=="V1"))
data3D$Item = data3D$Item + max(d.rs.V1$Item)+10
data3 = rbind(data3D %>% select(ModelPerformance, LogBeta, Surprisal, g, Model, Item, Language), d.rs.V1 %>% select(ModelPerformance, LogBeta, Surprisal, g, Model, Item, Language))
data3$german = ifelse(data3$Language == "german", 1, -1)
data3$LogBeta_ = data3$LogBeta
data3$LogBeta = data3$LogBeta - mean(data3$LogBeta)
data3$ModelPerformance = data3$ModelPerformance - mean(data3$ModelPerformance)

summary(fm3 <- lmer(Surprisal~ ModelPerformance*german*g +(1+g+german|Model)+(1+g|Item), data=data3)) # this is the effect on the verb
summary(fm3 <- lmer(Surprisal~ LogBeta*german*g +(1|Model)+(1+g|Item), data=data3))

library(ggplot2)
plot = ggplot(data3 %>% group_by(g, Model, Language, LogBeta) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=paste(Language,Model), color=Language)) + geom_line() + facet_wrap(~LogBeta)

crash()


data3 = data3 %>% mutate(Bottleneck = (!grepl("Control", Script))) %>% mutate(LogBeta = ifelse(Bottleneck, LogBeta, NA)) %>% mutate(grammatical = ifelse(g==-1, "Ungrammatical", "Grammatical"))
library(ggplot2)
plot = ggplot(data3 %>% group_by(grammatical, Model, LogBeta) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~LogBeta)
plot = ggplot(data3 %>% group_by(grammatical, Model, Memory) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Memory)
plot = ggplot(data3 %>% group_by(grammatical, Model, NumberOfDevRuns) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~NumberOfDevRuns)




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




