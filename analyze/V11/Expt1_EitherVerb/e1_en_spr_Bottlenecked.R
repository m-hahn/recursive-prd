## This file contains the non-final version of the data analyses for
##  Expt 1 presented in the paper:
#   Shravan Vasishth, Katja Suckow, Richard Lewis, and Sabine Kern.
#   Short-term forgetting in sentence comprehension:
#   Crosslinguistic evidence from head-final structures.
#   Submitted to Language and Cognitive Processes, 2007.

library(lme4)

library(Hmisc)
library(xtable)
library(MASS)

## interpretation:

## coefficient for gram:

## Process data for analysis:

## load data and prepare:

## reading time data analysis:
library(tidyr)
library(dplyr)
## preliminary data processing:
data <- read.csv("../../../stimuli/V11/English_EitherVerb.txt", sep="\t")
data$LineNumber = (1:nrow(data))-1

modelsTable = read.csv("~/scr/CODE/recursive-prd/results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("~/scr/CODE/recursive-prd/output/V11_E1_EitherVerb_english_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL)
datModel = merge(datModel, modelsTable, by=c("Model"))

data = merge(data, datModel, by=c("LineNumber"))

data = data %>% filter(!grepl("OOV", RegionLSTM))

data <- subset(data,expt=="gug")
data$expt <- factor(data$expt)

## make every position start with 1 instead of 0
data$position <- as.numeric(data$position)+1


library(reshape)

data = data %>% mutate(interference = ifelse(interference == "a", "hi", "lo"))
# TODO why is there an imbalance in this factor?

library(lme4)

summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1|Model)+(1|item), data=data %>% filter(region == "D1", grammatical!="full")))
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "V3", grammatical!="RemovedV1")))  # no difference between grammatical and ungrammatical
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "V3", grammatical!="RemovedV2"))) # no difference between grammatical and ungrammatical
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "V3", grammatical!="full"))) # no difference between either ungrammatical version
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "V2", grammatical!="RemovedV2"))) # removed V1 *better* than grammatical

# Both ungrammaticality manipulations show similar penalty in this surprisal.
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "D4", grammatical!="RemovedV1"))) # removedV2 *worse* than grammatical
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "D4", grammatical!="RemovedV2"))) # removedV1 *worse* than grammatical
summary(fm4a <- lmer(Surprisal~ grammatical+interference+(1+grammatical|Model)+(1+grammatical|item), data=data %>% filter(region == "D4", grammatical!="full"))) # no evidence for difference



