
library(tidyr)
library(dplyr)
library(lme4)

models = c(
905843526,
655887140,
766978233,
502504068,
697117799,
860606598
)

data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Staub2016_english_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv("../stimuli/Staub_2016/stims-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

raw.spr.data = raw.spr.data %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC"))
raw.spr.data = raw.spr.data %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE))
raw.spr.data = raw.spr.data %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))


orc_data = raw.spr.data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))

summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=raw.spr.data %>% filter(Region == "v1")))

