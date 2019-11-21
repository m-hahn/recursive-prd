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

data$True_Minus_False = data$True_False_False - data$False_False_False


data = data %>% mutate(True_Minus_False.C = True_Minus_False - mean(True_Minus_False, na.rm=TRUE))
data = data %>% mutate(Grammatical.C = Grammatical - mean(Grammatical, na.rm=TRUE))
data = data %>% mutate(ModelPerformance.C = ModelPerformance - mean(ModelPerformance, na.rm=TRUE))


summary(lmer(Surprisal ~ True_Minus_False.C * Grammatical.C * ModelPerformance.C + (1|Model) + (1+Grammatical.C|Noun) + (1|Remainder), data=data %>% filter(Region == "EOS")))


library(ggplot2)

data$Condition = as.factor(as.character(data$Condition))

plot = ggplot(data %>% filter(Region == c("EOS")) %>% group_by(Round, Model, Item, Condition, ModelPerformance, True_Minus_False) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Condition, True_Minus_False) %>% summarise(Surprisal=mean(Surprisal)), aes(x=True_Minus_False, y=Surprisal, group=Condition, color=Condition, fill=Condition)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~ModelPerformance)


plot = ggplot(data %>% filter(Region == c("EOS")) %>% group_by(Round, Model, Item, Condition, ModelPerformance, True_False_False) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Condition, True_False_False) %>% summarise(Surprisal=mean(Surprisal)), aes(x=True_False_False, y=Surprisal, group=Condition, color=Condition, fill=Condition)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~ModelPerformance)

###################################



