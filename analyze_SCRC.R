
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

modelData = data.frame()
for(model in models) {
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


summary(lmer(Surprisal ~ True_Minus_False.C * Grammatical.C + (1|Model) + (1+Grammatical.C|Noun) + (1 +True_Minus_False.C + Grammatical.C + True_Minus_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))

summary(lmer(Surprisal ~ True_False_False.C * Grammatical.C + (1|Model) + (1+Grammatical.C|Noun) + (1 +True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))

library(brms)
summary(brm(Surprisal ~ True_False_False.C * Grammatical.C + (1+True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Model) + (1+Grammatical.C|Noun) + (1 +True_False_False.C + Grammatical.C + True_False_False.C * Grammatical.C|Remainder), data=data %>% filter(Region == "EOS")))



library(ggplot2)

data$Condition = as.factor(as.character(data$Condition))

plot = ggplot(data %>% filter(Region == c("EOS")) %>% group_by(Round, Model, Item, Condition, True_Minus_False) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Model, Condition, True_Minus_False) %>% summarise(Surprisal=mean(Surprisal)), aes(x=True_Minus_False, y=Surprisal, group=Condition, color=Condition, fill=Condition)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~Model)


plot = ggplot(data %>% filter(Region == c("EOS")) %>% group_by(Round, Model, Item, Condition, Model, True_False_False) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Model, Condition, True_False_False) %>% summarise(Surprisal=mean(Surprisal)), aes(x=True_False_False, y=Surprisal, group=Condition, color=Condition, fill=Condition)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~Model)

###################################










data = data %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC"))
data = data %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE))
data = data %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))
data = data %>% mutate(ORC.C = (RCType == "ORC")-0.5)
data = data %>% mutate(Group = ifelse(HasParticle, "ORCPhrasal", ifelse(RCType == "ORC", "ORC", "SRC")), Length = ifelse(HasPP, "Long", "Short"))



library(ggplot2)


plot = ggplot(data %>% filter(Region %in% c("D1", "N1")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 

plot = ggplot(data %>% filter(!HasParticle, Region %in% c("V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(data %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 

library(lme4)

summary(lmer(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=data %>% filter(Region == "V0", !HasParticle)))

#library(brms)
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=data %>% filter(Region == "V0", !HasParticle)))



plot = ggplot(data %>% filter(Region %in% c("P0", "D2", "N2")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(data %>% filter(Region == "V1") %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
ggsave(plot, file="figures/staub2016_vanilla_v1.pdf", width=18, height=3.5)




orc_data = data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))


summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=data %>% filter(Region == "v1")))

