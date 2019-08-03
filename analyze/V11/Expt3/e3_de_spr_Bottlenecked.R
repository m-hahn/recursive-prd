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
data <- read.table("../../../../../recursive-prd/VSLK_LCP/E3_DE_SPR/data/e3desprdata.txt")
colnames(data) <- c("subj","expt","item","condition","position","word","RT","similarity","grammaticality")
data$LineNumber = (1:nrow(data))-1

modelsTable = read.csv("../../../results/models_bottlenecked_german", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("../../../output/V11_E3_german_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL)
datModel = merge(datModel, modelsTable, by=c("Model"))

data = merge(data, datModel, by=c("LineNumber"))

data = data %>% filter(!grepl("OOV", RegionLSTM))


NP3.data <- subset(data,position==5)
V3.data <- subset(data,position==6)
V2.data <- subset(data,(condition%in%c("a","b") & position==7))
V1.data <- subset(data,(condition%in%c("a","b") & position==8) | (condition%in%c("c","d") & position==7))
postV1.data <- subset(data,(condition%in%c("a","b") & position==9) |
                      (condition%in%c("c","d") & position==8))

d.NP3.rs <- melt(NP3.data, id=c("subj", "condition","item","word", "Model","Surprisal", "LogBeta", "Memory", "ModelPerformance", "Script"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.NP3.rs <- subset(d.NP3.rs,value>0)

d.NP3.rs$gram <- ifelse(d.NP3.rs$condition%in%c("a","b"),"gram","ungram")
d.NP3.rs$int <- ifelse(d.NP3.rs$condition%in%c("a","c"),"hi","lo")

d.V3.rs <- melt(V3.data, id=c("subj", "condition","item","word", "Model","Surprisal", "LogBeta", "Memory", "ModelPerformance", "Script"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.V3.rs <- subset(d.V3.rs,value>0)

d.V3.rs$gram <- ifelse(d.V3.rs$condition%in%c("a","b"),"gram","ungram")
d.V3.rs$int <- ifelse(d.V3.rs$condition%in%c("a","c"),"hi","lo")


d.V2.rs <- melt(V2.data, id=c("subj", "condition","item","word", "Model","Surprisal", "LogBeta", "Memory", "ModelPerformance", "Script"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.V2.rs <- subset(d.V2.rs,value>0)

d.V2.rs$gram <- ifelse(d.V2.rs$condition%in%c("a","b"),"gram",NA)
d.V2.rs$int <- ifelse(d.V2.rs$condition%in%c("a"),"hi","lo")

d.V1.rs <- melt(V1.data, id=c("subj", "condition","item","word", "Model","Surprisal", "LogBeta", "Memory", "ModelPerformance", "Script"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.V1.rs <- subset(d.V1.rs,value>0)

d.V1.rs$gram <- ifelse(d.V1.rs$condition%in%c("a","b"),"gram","ungram")
d.V1.rs$int <- ifelse(d.V1.rs$condition%in%c("a","c"),"hi","lo")

d.postV1.rs <- melt(postV1.data, id=c("subj", "condition","item","word", "Model","Surprisal", "LogBeta", "Memory", "ModelPerformance", "Script"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.postV1.rs <- subset(d.postV1.rs,value>0)

d.postV1.rs$gram <- ifelse(d.postV1.rs$condition%in%c("a","b"),"gram","ungram")
d.postV1.rs$int <- ifelse(d.postV1.rs$condition%in%c("a","c"),"hi","lo")


d.NP3.rs <- data.frame(region="NP3",d.NP3.rs)
d.V3.rs <- data.frame(region="V3",d.V3.rs)
d.V2.rs <- data.frame(region="V2",d.V2.rs)
d.V1.rs <- data.frame(region="V1",d.V1.rs)
d.postV1.rs <- data.frame(region="postV1",d.postV1.rs)

d.rs <- rbind(d.NP3.rs,d.V3.rs,d.V2.rs,d.V1.rs,d.postV1.rs)

head(d.rs)
summary(d.rs)

#centered length
#d.V.rs$length <- center(nchar(as.character(d.V.rs$word)))

critdata <- d.rs

g <- ifelse(critdata$condition%in%c("a","b"),1,-1)
i <- ifelse(critdata$condition%in%c("a","c"),1,-1)
gxi <- ifelse(critdata$condition%in%c("a","d"),1,-1)

critdata$g <- g
critdata$i <- i
critdata$gxi <- gxi


summary(critdata)

## comparison 0:
data0 <- subset(critdata,region=="NP3")

summary(fm0 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|item),
                   data=data0))

## comparison 1:
data1 <- subset(critdata,region=="V3")

summary(fm1 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|item),
                   data=data1))


## comparison 2:

data2 <- subset(critdata,(region=="V2" & condition%in%c("a","b")) |
                                        (region=="V1" & condition%in%c("c","d")))

summary(fm2 <- lmer(Surprisal~ g+i+gxi+
                    #center(wl)+
                    (1|Model),#+(1|item),
                    data=data2))



## comparison 3:
data3 <- subset(critdata,(region=="V1"))

summary(fm3 <- lmer(Surprisal~ g+i+gxi +(1|Model)+(1|item), data=data3)) # this is the effect on the verb



data3 = data3 %>% mutate(LogBeta.C = LogBeta-mean(LogBeta))
data3 = data3 %>% mutate(ModelPerformance.C = ModelPerformance-mean(ModelPerformance))

#summary(fm3 <- lmer(Surprisal~ g+i+gxi+(1|Model)+(1|item), data=data3))
#summary(fm3 <- lmer(Surprisal~ ModelPerformance.C*g+i+gxi+(1+g+i+gxi|Model)+(1+ModelPerformance.C+g+i+gxi|item), data=data3)) # worse models generate a larger forgetting effect

data3 = data3 %>% mutate(Bottleneck = (!grepl("Control", Script))) %>% mutate(LogBeta = ifelse(Bottleneck, LogBeta, NA)) %>% mutate(grammatical = ifelse(g==-1, "Ungrammatical", "Grammatical"))
library(ggplot2)
plot = ggplot(data3 %>% group_by(grammatical, Model, LogBeta) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~LogBeta)
#plot = ggplot(data3 %>% group_by(grammatical) %>% summarise(Surprisal=mean(Surprisal)), aes(x=1, y=Surprisal, group=grammatical, fill=grammatical)) + geom_bar(stat="identity", position=position_dodge(0.9))


# TODO something is wrong: e3_de_spr.R records more on-OOV words (e.g., ueberzeugte, begruesste). Maybe the Python 2.7 code doesn't deal correctly with umlauts in the input??!! (but the bottlenecked language model training code doesn't show this problem)
# In any case, there are too many OOV items to say anything.


## comparison 4:
data4 <- subset(critdata,(region=="postV1"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1|Model)+(1|item),
                   data=data4))

summary(fm4 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4)))

summary(fm4 <- lmer(value~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4)))

## this model is better
summary(fm4a <- lmer(log(value)~ g+i+gxi+(1+g|subject)+(1|item),
                   data=subset(data4)))

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
#plot = ggplot(data4 %>% group_by(grammatical) %>% summarise(Surprisal=mean(Surprisal)), aes(x=grammatical, y=Surprisal)) + geom_line() #(stat="identity", position=position_dodge(0.9))



