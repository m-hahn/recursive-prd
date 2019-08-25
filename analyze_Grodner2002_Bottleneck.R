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

# No evidence for any effect
summary(lmer(Surprisal ~ ModelPerformance.C * RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1|Model), data=dataCriticalVerb))


summary(lmer(Surprisal ~ RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1 + RC + Ambiguous + RC * Ambiguous|Model), data=dataCriticalVerb))


###############################3

# main ROI in the paper: disambiguating By ohrase

dataBy = raw.spr.data %>% filter(Region %in% c("P1", "D3", "N3")) %>% group_by(Condition1, Condition2, Item, Model, Round, RC, Ambiguous, ModelPerformance.C) %>% summarise(Surprisal = sum(Surprisal))

# Better models have a stronger RC-SC effect.
summary(lmer(Surprisal ~ ModelPerformance.C * RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1|Model), data=dataBy))



summary(lmer(Surprisal ~ RC * Ambiguous + (1 + RC + Ambiguous + RC * Ambiguous|Item) + (1 + RC + Ambiguous + RC * Ambiguous|Model), data=dataBy))



library(ggplot2)


#######################################

plot = ggplot(raw.spr.data %>% filter(Region %in% c("D1", "N1")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 

plot = ggplot(raw.spr.data %>% filter(!HasParticle, Region %in% c("V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(raw.spr.data %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


#library(brms)
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "V0", !HasParticle)))



plot = ggplot(raw.spr.data %>% filter(Region %in% c("P0", "D2", "N2")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(raw.spr.data %>% filter(Region == "V1") %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
ggsave(plot, file="figures/staub2016_vanilla_v1.pdf", width=18, height=3.5)




orc_data = raw.spr.data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))


summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=raw.spr.data %>% filter(Region == "v1")))

