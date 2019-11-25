# Other method of testing: double object construction
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
section = "_explore21"
# 3 with tried to: same effect
# 8 shows that same effect happens with simple present
# 10 with was (passive):
data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Staub2016", section,"_english_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv(paste("../stimuli/Staub_2016/stims-tokenized", section, ".tsv", sep=""), sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1
 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))
mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))
raw.spr.data = raw.spr.data %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC"))
raw.spr.data = raw.spr.data %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE))
raw.spr.data = raw.spr.data %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))
raw.spr.data = raw.spr.data %>% mutate(ORC.C = (RCType == "ORC")-0.5)
raw.spr.data = raw.spr.data %>% mutate(Group = ifelse(HasParticle, "ORCPhrasal", ifelse(RCType == "ORC", "ORC", "SRC")), Length = ifelse(HasPP, "Long", "Short"))
library(ggplot2)
plot = ggplot(raw.spr.data %>% filter(Region %in% c("D1", "N1")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
plot = ggplot(raw.spr.data %>% filter(!HasParticle, Region %in% c("V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
plot = ggplot(raw.spr.data %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
library(lme4)
#summary(lmer(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "V0", !HasParticle)))
#library(brms)
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "V0", !HasParticle)))
plot = ggplot(raw.spr.data %>% filter(Region %in% c("P0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
plot = ggplot(raw.spr.data %>% filter(Region %in% c("P0", "DN2")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 

plot = ggplot(raw.spr.data %>% filter(Region == "V1") %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
ggsave(plot, file=paste("figures/staub2016_vanilla_v1", section, ".pdf", sep=""), width=18, height=3.5)




orc_data = raw.spr.data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))


summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=raw.spr.data %>% filter(Region == "v1")))

