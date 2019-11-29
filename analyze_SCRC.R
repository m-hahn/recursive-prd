
library(tidyr)
library(dplyr)
library(lme4)

models_data = read.csv("results/models_vanillaLSTM_english.tsv", sep="\t")

modelData = data.frame()
for(model in models_data$Model) {
   modelData2 = read.csv(paste("output/SCRC_default_english_", model, sep=""), sep="\t")
   modelData2$Model = model
   modelData = rbind(modelData, modelData2)
}
modelData = modelData %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 data <- read.csv("../stimuli/SC_RC_lexical/stimuli-tokenized.tsv", sep="\t")
 data$LineNumber = (1:nrow(data))-1

 data = merge(data, modelData, by=c("LineNumber"))

mean(as.character(data$Word) == as.character(data$RegionLSTM))

data = data %>% mutate("Grammatical" = (Condition == 0))
data = data %>% mutate("DroppedMiddleVerb" = (Condition == 2))

data = merge(data, models_data, by=c("Model"), all.x=TRUE)


nounFreqs = read.csv("../forgetting/corpus/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)
nounFreqs2 = read.csv("../forgetting/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)
nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]
data = merge(data, nounFreqs %>% rename(Noun=noun), by=c("Noun"), all.x=TRUE)

data$True_Minus_False = data$True_False_False - data$False_False_False

data = data %>% mutate(True_False_False.C = True_False_False - mean(True_False_False, na.rm=TRUE))
data = data %>% mutate(True_Minus_False.C = True_Minus_False - mean(True_Minus_False, na.rm=TRUE))
data = data %>% mutate(Grammatical.C = Grammatical - mean(Grammatical, na.rm=TRUE))

summary(lmer(Surprisal ~ True_Minus_False.C * Grammatical.C + (1+True_Minus_False.C + Grammatical.C + True_Minus_False.C * Grammatical.C|Model) + (1+Grammatical.C|Noun) + (1 +True_Minus_False.C + Grammatical.C + True_Minus_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))

summary(lmer(Surprisal ~ True_False_False.C * Grammatical.C + (1+True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Model) + (1+Grammatical.C|Noun) + (1 +True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))

library(brms)

model_TM = (brm(Surprisal ~ True_Minus_False.C * Grammatical.C + (1+True_Minus_False.C + Grammatical.C + True_Minus_False.C * Grammatical.C|Model) + (1+Grammatical.C|Noun) + (1 +True_Minus_False.C + Grammatical.C + True_Minus_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))


model_TF = (brm(Surprisal ~ True_False_False.C * Grammatical.C + (1+True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Model) + (1+Grammatical.C|Noun) + (1 +True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))



library(ggplot2)

data$Condition = as.factor(as.character(data$Condition))

plot = ggplot(data %>% filter(Region == c("EOS")) %>% group_by(Round, Model, Item, Condition, True_Minus_False) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Model, Condition, True_Minus_False) %>% summarise(Surprisal=mean(Surprisal)), aes(x=True_Minus_False, y=Surprisal, group=Condition, color=Condition, fill=Condition)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~Model)
ggsave(plot, file="figures/SCRC_EOSSurpByLogFreqRatio.pdf")

plot = ggplot(data %>% filter(Region == c("EOS")) %>% group_by(Round, Model, Item, Condition, Model, True_False_False) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Model, Condition, True_False_False) %>% summarise(Surprisal=mean(Surprisal)), aes(x=True_False_False, y=Surprisal, group=Condition, color=Condition, fill=Condition)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~Model)

###################################


surprisals = data %>% group_by(Model, True_Minus_False, Condition) %>% summarise(Surprisal=mean(Surprisal))





