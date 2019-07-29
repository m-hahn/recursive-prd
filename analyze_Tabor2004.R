
library(tidyr)
library(dplyr)
library(lme4)

models = c(
905843526,
655887140,
766978233,
502504068,
697117799,
860606598)

data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Tabor2004_english_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv("../stimuli/tabor_2004/expt1_3_tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

raw.spr.data = raw.spr.data %>% mutate(Reduced = (Condition %in% c("c", "d"))-0.5, AmbiguousVerb = ((Condition %in% c("a", "c")) - 0.5))


summary(lmer(Surprisal ~ Reduced*AmbiguousVerb + (1+Reduced+AmbiguousVerb+Reduced*AmbiguousVerb|Item) + (1+Reduced+AmbiguousVerb+Reduced*AmbiguousVerb|Model), data=raw.spr.data %>% filter(Region == "object")))

summary(lmer(Surprisal ~ Reduced*AmbiguousVerb + (1+Reduced+AmbiguousVerb+Reduced*AmbiguousVerb|Item) + (1+Reduced+AmbiguousVerb+Reduced*AmbiguousVerb|Model), data=raw.spr.data %>% filter(Region == "participle")))

summary(lmer(Surprisal ~ Reduced*AmbiguousVerb + (1+Reduced+AmbiguousVerb+Reduced*AmbiguousVerb|Item) + (1+Reduced+AmbiguousVerb+Reduced*AmbiguousVerb|Model), data=raw.spr.data %>% filter(Region == "by_P")))






