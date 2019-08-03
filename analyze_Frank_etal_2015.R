
library(tidyr)
library(dplyr)
library(lme4)

models = c(
785335595,
709209753
)

data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Frank_etal_2015_dutch_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv("../stimuli/Frank_Trompenaars_Vasishth_2015/dutch_tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

raw.spr.data = raw.spr.data %>% mutate(Grammatical.C = (Condition == "g")-0.5)

summary(lmer(Surprisal ~ Grammatical.C + (1+Grammatical.C|Item) + (1+Grammatical.C|Model), data=raw.spr.data %>% filter(Region == "v2")))
# This model shows a forgetting effect (unexpectedly!)
summary(lmer(Surprisal ~ Grammatical.C + (1+Grammatical.C|Item) + (1+Grammatical.C|Model), data=raw.spr.data %>% filter(Region == "d3")))


