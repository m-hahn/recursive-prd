library(tidyr)
library(dplyr)
library(lme4)

 data <- read.csv("../stimuli/SC_RC_lexical/stimuli-tokenized.tsv", sep="\t")
 data$LineNumber = (1:nrow(data))-1

modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/SCRC_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL, Bottlenecked = !grepl("Control", Script), LogBeta = ifelse(Bottlenecked, LogBeta, NA))
datModel = merge(datModel, modelsTable, by=c("Model"))
datModel = datModel %>% filter(RegionLSTM != "OOV")

 data = merge(data, datModel, by=c("LineNumber"))

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


library(ggplot2)


plot = ggplot(data %>% filter(Region %in% c("D1", "N1")) %>% group_by(Round, Model, Item, Condition, ModelPerformance, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~ModelPerformance)

plot = ggplot(data %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, ModelPerformance, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~ModelPerformance)


plot = ggplot(data %>% filter(Region %in% c("P0", "D2", "N2")) %>% group_by(Round, Model, Item, Condition, ModelPerformance, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


# Surprisal on the Embedded Verb
plot = ggplot(data %>% filter(Region == "V0", Group != "ORCPhrasal") %>% group_by(ModelPerformance, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal)) + geom_point() + facet_wrap(~ModelPerformance)
ggsave(plot, file="figures/staub2016_bottlenecked_v0.pdf", width=18, height=3.5)



# Surprisal on the Matrix Verb
plot = ggplot(data %>% filter(Region == "V1") %>% group_by(ModelPerformance, Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) + facet_grid(~ModelPerformance)
ggsave(plot, file="figures/staub2016_bottlenecked_v1.pdf", width=18, height=3.5)




orc_data = data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))

orc_data = orc_data %>% mutate(LogBeta.C = LogBeta - mean(LogBeta, na.rm=TRUE))

summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=data %>% filter(Region == "v1")))


