library(tidyr)
library(dplyr)
library(lme4)


modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/Traxler2002_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL, Bottlenecked = !grepl("Control", Script), LogBeta = ifelse(Bottlenecked, LogBeta, NA))
datModel = merge(datModel, modelsTable, by=c("Model"))
datModel = datModel %>% filter(RegionLSTM != "OOV")


 raw.spr.data <- read.csv("../stimuli/traxler_etal_2002/expt1-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1


 raw.spr.data = merge(raw.spr.data, datModel, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

raw.spr.data = raw.spr.data %>% mutate(ORC.C=(Condition == "ORC")-0.5, LogBeta.C = LogBeta-mean(LogBeta), ModelPerformance.C=ModelPerformance-mean(ModelPerformance))


summary(lmer(Surprisal ~ ORC.C*LogBeta.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v0")))

summary(lmer(Surprisal ~ ORC.C*LogBeta.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v1")))

library(brms)


#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v0"), chains=1))
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v1"), chains=1))


relcl = raw.spr.data %>% filter(Region %in% c("d1", "n1", "v0")) %>% group_by(LogBeta.C, Model, Item, ORC.C) %>% summarise(Surprisal=mean(Surprisal))
summary(lmer(Surprisal ~ LogBeta.C*ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=relcl))





