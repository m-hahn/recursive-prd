library(tidyr)
library(dplyr)
library(lme4)

 raw.spr.data <- read.csv("../stimuli/Staub_2016/stims-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/Staub2016_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
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

raw.spr.data = raw.spr.data %>% mutate("RCType" = ifelse(Condition %in% c("A", "B", "C", "D"), "ORC", "SRC"))
raw.spr.data = raw.spr.data %>% mutate("HasPP" = ifelse(Condition %in% c("B", "D", "F"), TRUE, FALSE))
raw.spr.data = raw.spr.data %>% mutate("HasParticle" = ifelse(Condition %in% c("C", "D"), TRUE, FALSE))

raw.spr.data = raw.spr.data %>% mutate(Group = ifelse(HasParticle, "ORCPhrasal", ifelse(RCType == "ORC", "ORC", "SRC")), Length = ifelse(HasPP, "Long", "Short"))



library(ggplot2)


plot = ggplot(raw.spr.data %>% filter(Region %in% c("D1", "N1")) %>% group_by(Round, Model, Item, Condition, LogBeta, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(LogBeta, Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~LogBeta)

plot = ggplot(raw.spr.data %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, LogBeta, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(LogBeta, Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~LogBeta)


plot = ggplot(raw.spr.data %>% filter(Region %in% c("P0", "D2", "N2")) %>% group_by(Round, Model, Item, Condition, LogBeta, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(raw.spr.data %>% filter(Region == "V1") %>% group_by(LogBeta, Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) + facet_grid(~LogBeta)
ggsave(plot, file="figures/staub2016_bottlenecked_v1.pdf", width=18, height=3.5)




orc_data = raw.spr.data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))

orc_data = orc_data %>% mutate(LogBeta.C = LogBeta - mean(LogBeta, na.rm=TRUE))

summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=raw.spr.data %>% filter(Region == "v1")))


