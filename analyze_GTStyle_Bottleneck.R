library(tidyr)
library(dplyr)
library(lme4)

# raw.spr.data <- read.csv("../stimuli/Gibson_Thomas_style/tokenized.tsv", sep="\t")
 raw.spr.data <- read.csv("../stimuli/Frank2019/tokenized.tsv", sep="\t")

 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/GTStyle_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
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



###################################################################################################################



library(ggplot2)

raw.spr.data$Region = factor(raw.spr.data$Region, levels=c("D1", "N1", "THAT1", "D2", "N2", "THAT2", "D3", "N3", "V3", "V2", "V1", "Punct"))

plot = ggplot(raw.spr.data %>% filter(ModelPerformance < 5) %>% group_by(Region, Round, Model, Item, Condition, ModelPerformance) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Region, ModelPerformance, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Region, y=Surprisal, group=paste(Condition), color=paste(Condition), fill=paste(Condition)))
plot = plot + geom_line()  + facet_wrap(~ModelPerformance)



plot = ggplot(raw.spr.data %>% filter(Region %in% c("Punct")) %>% group_by(Round, Model, Item, Condition, ModelPerformance) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=1, y=Surprisal, group=paste(Condition), color=paste(Condition), fill=paste(Condition))) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~ModelPerformance)


plot = ggplot(raw.spr.data %>% filter(Region %in% c("V2")) %>% group_by(Round, Model, Item, Condition, ModelPerformance) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=1, y=Surprisal, group=paste(Condition), color=paste(Condition), fill=paste(Condition))) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~ModelPerformance)


plot = ggplot(raw.spr.data %>% filter(Region %in% c("V1")) %>% group_by(Round, Model, Item, Condition, ModelPerformance) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(ModelPerformance, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=1, y=Surprisal, group=paste(Condition), color=paste(Condition), fill=paste(Condition))) + geom_bar(stat="identity", position=position_dodge(width=0.9))  + facet_grid(~ModelPerformance)



library(lme4)
raw.spr.data = raw.spr.data %>% mutate(ModelPerformance.C = ModelPerformance - mean(ModelPerformance))
raw.spr.data = raw.spr.data %>% mutate(NoV3.C = (Condition == "NoV3") - mean((Condition == "NoV3")))



summary(lmer(Surprisal ~ NoV3.C*ModelPerformance.C + (NoV3.C|Model) + (NoV3.C|Item), data= raw.spr.data%>% filter(Region %in% c("V2"))))


################################



