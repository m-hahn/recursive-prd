library(tidyr)
library(dplyr)
library(lme4)

 raw.spr.data <- read.csv("../stimuli/Chen_etal_2005/expt1-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/Chen2005_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
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

raw.spr.data = raw.spr.data %>% mutate(Verb1 = ifelse(Condition %in% c("zero", "one_late"), 0.5, -0.5))

raw.spr.data = raw.spr.data %>% mutate(Verb2 = ifelse(Condition %in% c("zero", "one_early"), 0.5, -0.5))

raw.spr.data = raw.spr.data %>% mutate(ModelPerformance.C = ModelPerformance-mean(ModelPerformance))

############################3



summary(lmer(Surprisal ~ ModelPerformance.C*Verb1*Verb2 + (1 + Verb1+Verb2 + Verb1*Verb2 | Item) + (1+Verb1+Verb2+Verb1*Verb2|Model), data=raw.spr.data %>% filter(Region == "critical")))
# Storage cost effects, also an interaction

library(ggplot2)
plot = ggplot(data=raw.spr.data %>% filter(Region == "critical") %>% group_by(ModelPerformance, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal))
plot = plot + geom_point()
plot = plot + facet_wrap(~ModelPerformance)
ggsave(plot, file="figures/chen2005-expt1-critical.pdf")



