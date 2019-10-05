library(tidyr)
library(dplyr)
library(lme4)

 raw.spr.data <- read.csv("../stimuli/Grodner_etal_2002/expt1-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/Grodner2002_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL, Bottlenecked = !grepl("Control", Script), LogBeta = ifelse(Bottlenecked, LogBeta, NA))
datModel = merge(datModel, modelsTable, by=c("Model"))
datModel = datModel %>% filter(RegionLSTM != "OOV")

 raw.spr.data = merge(raw.spr.data, datModel, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

library(lme4)

raw.spr.data = raw.spr.data %>% mutate(RC = ifelse(Condition1 == "RC", 0.5, -0.5))
raw.spr.data = raw.spr.data %>% mutate(Ambiguous = ifelse(Condition2 == "ambiguous", 0.5, -0.5))
raw.spr.data = raw.spr.data %>% mutate(ModelPerformance.C = ModelPerformance-mean(ModelPerformance))

############################3

dataNextThree1 = raw.spr.data %>% filter(grepl("ThreeWords1", Region)) %>% group_by(Condition1, Condition2, Item, Model, Round, RC, Ambiguous, ModelPerformance.C) %>% summarise(Surprisal = sum(Surprisal))


# Better models show a bigger RC-SC effect
summary(lmer(Surprisal ~ ModelPerformance.C*RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1|Model), data=dataNextThree1))


summary(lmer(Surprisal ~ RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1 + RC + Ambiguous + RC * Ambiguous|Model), data=dataNextThree1))

###############################3


dataNextThree = raw.spr.data %>% filter(grepl("ThreeWords", Region)) %>% group_by(Condition1, Condition2, Item, Model, Round, RC, Ambiguous, ModelPerformance.C) %>% summarise(Surprisal = sum(Surprisal))


# Better models show a bigger RC-SC effect
summary(lmer(Surprisal ~ ModelPerformance.C*RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1|Model), data=dataNextThree))


summary(lmer(Surprisal ~ RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1 + RC + Ambiguous + RC * Ambiguous|Model), data=dataNextThree))

###############################3

dataCriticalVerb = raw.spr.data %>% filter(Region == ifelse(Condition1 == "RC", "V1", "V2"))

# No evidence for any effect (but there are differences, see plot)
summary(lmer(Surprisal ~ ModelPerformance.C * RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1|Model), data=dataCriticalVerb))


summary(lmer(Surprisal ~ RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1 + RC + Ambiguous + RC * Ambiguous|Model), data=dataCriticalVerb))

# TODO really selected correctly?

library(ggplot2)

plot = ggplot(data=dataCriticalVerb %>% group_by(Condition1, Condition2, ModelPerformance) %>% summarise(Surprisal=mean(Surprisal)), aes(x=paste(Condition1, Condition2), y=Surprisal)) + geom_point() + facet_wrap(~ModelPerformance)

###############################3

# main ROI in the paper: disambiguating By ohrase

dataBy = raw.spr.data %>% filter(Region %in% c("P1", "D3", "N3")) %>% group_by(Condition1, Condition2, Item, Model, Round, RC, Ambiguous, ModelPerformance.C, ModelPerformance) %>% summarise(Surprisal = sum(Surprisal))

plot = ggplot(data=dataBy %>% group_by(Condition1, Condition2, ModelPerformance) %>% summarise(Surprisal=mean(Surprisal)), aes(x=paste(Condition1, Condition2), y=Surprisal)) + geom_point() + facet_wrap(~ModelPerformance)



# Better models have a stronger RC-SC effect.
summary(lmer(Surprisal ~ ModelPerformance.C * RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1|Model), data=dataBy))
# Interaction in the other direction than in the human data


summary(lmer(Surprisal ~ RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1 + RC + Ambiguous + RC * Ambiguous|Model), data=dataBy))


