library(tidyr)
library(dplyr)
library(lme4)


modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/Traxler2002_expt1_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL, Bottlenecked = !grepl("Control", Script), LogBeta = ifelse(Bottlenecked, LogBeta, NA))
datModel = merge(datModel, modelsTable, by=c("Model"))
datModel = datModel %>% filter(RegionLSTM != "OOV")


 raw.spr.data <- read.csv("../stimuli/traxler_etal_2002/expt1-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1


 raw.spr.data = merge(raw.spr.data, datModel, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

raw.spr.data = raw.spr.data %>% mutate(ORC.C=(Condition == "ORC")-0.5, LogBeta.C = LogBeta-mean(LogBeta, na.rm=TRUE), ModelPerformance.C=ModelPerformance-mean(ModelPerformance, na.rm=TRUE))

# TODO Figure out (using more models) whether this one is modulated by LogBeta
summary(lmer(Surprisal ~ ORC.C*LogBeta.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v0")))

summary(lmer(Surprisal ~ ORC.C*LogBeta.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v1")))

library(brms)


#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v0"), chains=1))
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v1"), chains=1))


relcl = raw.spr.data %>% filter(Region %in% c("d1", "n1", "v0")) %>% group_by(LogBeta.C, Model, Item, ORC.C) %>% summarise(Surprisal=mean(Surprisal))
summary(lmer(Surprisal ~ LogBeta.C*ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=relcl))

# Total surprisal on embedded NP
np = raw.spr.data %>% filter(Region %in% c("d1", "n1")) %>% group_by(LogBeta.C, Model, Item, ORC.C, Round) %>% summarise(Surprisal=sum(Surprisal))
summary(lmer(Surprisal ~ LogBeta.C*ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=np))



raw.spr.data = raw.spr.data %>% mutate(RegPosition = case_when(Region == "d0" ~ 1, Region == "n0" ~ 2, Region == "c" ~ 3, Region == "d1" & Condition == "ORC" ~ 4, Region == "n1" & Condition == "ORC" ~ 5, Region == "v0" & Condition == "ORC" ~ 6, Region == "v1" ~ 7, Region == "post" ~ 8, Region == "v0" & Condition == "SRC" ~ 4, Region == "d1" & Condition == "SRC" ~ 5, Region == "n1" & Condition == "SRC" ~ 6))
raw.spr.data = raw.spr.data %>% mutate(RegWord = case_when(Region == "d0" ~ "the", Region == "n0" ~ "banker", Region == "c" ~ "that", Region == "d1" ~ "the", Region == "n1" ~ "lawyer", Region == "v0" ~ "irritated", Region == "v1" ~ "played", Region == "post" ~ "tennis every Sunday"))

library(ggplot2)
plot = ggplot(raw.spr.data %>% group_by(ModelPerformance, RegWord, RegPosition, Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=RegPosition, y=Surprisal, group=Condition, color=Condition))
plot = plot + geom_line()
plot = plot + geom_text(aes(label=RegWord))
plot = plot + facet_wrap(~ModelPerformance)
ggsave(plot, file="traxler2002-full-bottlenecked.pdf", height=15, width=15)

library(ggplot2)
plot = ggplot(raw.spr.data %>% filter(Region == "v0") %>% group_by(LogBeta, RegWord, RegPosition, Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=RegPosition, y=Surprisal, group=Condition, color=Condition))
plot = plot + geom_line()
plot = plot + geom_text(aes(label=RegWord))
plot = plot + facet_wrap(~LogBeta)

# This looks like convincing evidence for a U-shaped curve
plot = ggplot(raw.spr.data %>% filter(Region == "v0") %>% group_by(ModelPerformance, RegWord, RegPosition, Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=RegPosition, y=Surprisal, group=Condition, color=Condition))
plot = plot + geom_line()
plot = plot + geom_text(aes(label=RegWord))
plot = plot + facet_wrap(~ModelPerformance)
ggsave(plot, file="traxler2002-embeddedVerb-bottlenecked.pdf")

