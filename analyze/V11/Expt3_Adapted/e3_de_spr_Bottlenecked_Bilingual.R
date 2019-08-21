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
dataD <- read.csv("../../../../stimuli/V11/V11_German_adapted_tokenized.txt", sep="\t")
dataD$LineNumber = (1:nrow(dataD))-1
dataD$expt = 'de' 
dataD$Round = NULL
dataD$Language = 'german'
dataD$position = 0

dataE <- read.table("../../../../../recursive-prd/VSLK_LCP/E1_EN_SPR/data/e1_en_spr_data.txt")
colnames(dataE) <- c("subject","expt","Item","Condition","position","Word","rt")
dataE$LineNumber = (1:nrow(dataE))-1
dataE$subject = NULL
dataE$rt = NULL
dataE$Region = "NoRegion"
dataE$Language = "english"

data = rbind(dataD, dataE)

modelsTable = read.csv("../../../results/models_bottlenecked_bilingual", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(language in c("english", "german")) {
  if(language == "english") {
       exp = "E1"
  } else {
       exp = "E3_Adapted"
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

dataD$g <- ifelse(dataD$Condition%in%c("a","b"),1,-1)
dataD$i <- ifelse(dataD$Condition%in%c("a","c"),1,-1)
dataD$gxi <- ifelse(dataD$Condition%in%c("a","d"),1,-1)

######################################################

dataE = data %>% filter(Language == "english")

dataE <- subset(dataE,expt=="gug")
dataE$expt <- factor(dataE$expt)

## make every position start with 1 instead of 0
dataE$position <- as.numeric(dataE$position)+1


dataE$gram <- ifelse(dataE$Condition%in%c("a","b"),"gram","ungram")
dataE$int <- ifelse(dataE$Condition%in%c("a","c"),"hi","lo")


library(reshape)

## computations cached:
d.rs <- dataE

unique(subset(d.rs,position==1)$word)


## recode regions of interest:

## The painter who the film that the friend liked admired the poet.  
##  1    2      3   4   5   6     7   8     9     10      11  12
##    1         2     3     4       5 

## recode the regions of interest:
d.rs$roi <- ifelse(d.rs$position==2,1, # NP1
               ifelse(d.rs$position==3,2, # who
                 ifelse(d.rs$position%in%c(4,5),3, #NP2
                            ifelse(d.rs$position==6,4, # that
                                   ifelse(d.rs$position%in%c(7,8),5, # NP3
                                                 d.rs$position)))))



dataE.ab <- subset(dataE,Condition%in%c("a","b"))
dataE.cd <- subset(dataE,Condition%in%c("c","d"))

pos10dataE <- subset(dataE.ab,position==10) ## V2 in a,b

d.rs.V2ab <- pos10dataE

d.rs.V2ab$gram <- ifelse(d.rs.V2ab$Condition%in%c("a","b"),"gram","ungram")

d.rs.V2ab$int <- ifelse(d.rs.V2ab$Condition%in%c("a"),"hi",
                        ifelse(d.rs.V2ab$Condition%in%c("b"),"lo",NA))


## V1:

pos11dataE <- subset(dataE.ab,position==11)
pos10dataE <- subset(dataE.cd,position==10)
pos1011dataE <- rbind(pos10dataE,pos11dataE)

d.rs.V1 <- pos1011dataE



d.rs.V1$g <- ifelse(d.rs.V1$Condition%in%c("a","b"),1,-1)
d.rs.V1$i <- ifelse(d.rs.V1$Condition%in%c("a","c"),1,-1)
d.rs.V1$gxi <- ifelse(d.rs.V1$Condition%in%c("a","d"),1,-1)


## comparison 3:
data3D <- subset(dataD,(Region=="V1"))
data3D$Item = data3D$Item + max(d.rs.V1$Item)+10
data3 = rbind(data3D %>% select(LogBeta, Surprisal, g, i, gxi, Model, Item, Language), d.rs.V1 %>% select(LogBeta, Surprisal, g, i, gxi, Model, Item, Language))
data3$german = ifelse(data3$Language == "german", 1, -1)
data3$LogBeta = data3$LogBeta - mean(data3$LogBeta)

summary(fm3 <- lmer(Surprisal~ german*g+i+gxi +(1+g+german|Model)+(1+g|Item), data=data3)) # this is the effect on the verb


library(ggplot2)
plot = ggplot(data3 %>% group_by(g, Model, Item) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Item)


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




