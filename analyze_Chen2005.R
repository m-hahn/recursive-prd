
library(tidyr)
library(dplyr)
library(lme4)


models_data = read.csv("results/models_vanillaLSTM_english.tsv", sep="\t") #, header=c("Model", "Script", "ModelPerformance"))

data = data.frame()
for(model in models_data$Model) {
   data2 = read.csv(paste("output/Chen2005_english_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv("../stimuli/Chen_etal_2005/expt1-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))


library(lme4)

raw.spr.data = raw.spr.data %>% mutate(Verb1 = ifelse(Condition %in% c("zero", "one_late"), 0.5, -0.5))

raw.spr.data = raw.spr.data %>% mutate(Verb2 = ifelse(Condition %in% c("zero", "one_early"), 0.5, -0.5))


############################3



summary(lmer(Surprisal ~ Verb1*Verb2 + (1 + Verb1+Verb2 + Verb1*Verb2 | Item) + (1+Verb1+Verb2+Verb1*Verb2|Model), data=raw.spr.data %>% filter(Region == "critical")))
# Storage cost effects, also an interaction


