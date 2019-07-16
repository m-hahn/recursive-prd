models = c(
73547463     , 
914171046     ,
47228214      ,
371571241     ,
858779689     ,
963791434     ,
25588231      ,
710633506     ,
392207746     ,
938059340)

library(dplyr)
library(tidyr)
library(lme4)

dataBase = read.csv("stimuli/husain_etal_2014_hindi/final_items_all_tokenized.txt", sep="\t")
dataBase$LineNumber = (1:nrow(dataBase))

data = data.frame()
for(model in models) {
   data2 = read.csv("output/HVS_371571241", sep="\t") %>% filter(!grepl("OOV", RegionLSTM)) %>% mutate(Model = model)
   data = rbind(data, data2)
}
dataBase = merge(data, dataBase)

###############################

# Now undo the tokenization

dataBase = dataBase %>% group_by(Item, Model, Condition, Experiment, Region, OriginalTokenNumber) %>% summarise(Surprisal=sum(Surprisal))

###############################

# This is Experiment 1 in the Paper

exp1 = dataBase %>% filter(Experiment == "RC1") %>% mutate(SubjRC = (Condition %in% c("a", "b")) - 0.5, Long = (Condition %in% c("a", "c")) - 0.5)
summary(lmer(Surprisal ~ SubjRC*Long + (1+SubjRC+Long+SubjRC*Long|Item) + (1+SubjRC+Long+SubjRC*Long|Model), exp1 %>% filter(Region == "RCVerb")))

###############################

# TODO something is weird, the Expectation > 0 critical surprisals are all around 16?!
exp2 = dataBase %>% filter(Experiment == "RC2") %>% mutate(Expectation = (Condition %in% c("a", "b")) - 0.5, Long = (Condition %in% c("b", "d")) - 0.5)
summary(lmer(Surprisal ~ Expectation*Long + (1+Expectation+Long+Expectation*Long|Item) + (1+Expectation+Long+Expectation*Long|Model), exp2 %>% filter((Expectation > 0 && Region == "NP1") || (Expectation < 0 && Region == "RCPn"))))

###############################

exp3 = dataBase %>% filter(Experiment == "PP1") %>% mutate(Expectation = (Condition %in% c("b", "d")) - 0.5, Long = (Condition %in% c("a", "b")) - 0.5)
summary(lmer(Surprisal ~ Expectation*Long + (1+Expectation+Long+Expectation*Long|Item) + (1+Expectation+Long+Expectation*Long|Model), exp3 %>% filter(Region == "HeadNP")))


###############################

# This is Experiment 2 in the Paper

exp4 = dataBase %>% filter(Experiment == "CP1") %>% mutate(Expectation = (Condition %in% c("a", "b")) - 0.5, Long = (Condition %in% c("a", "c")) - 0.5)
summary(lmer(Surprisal ~ Expectation*Long + (1+Expectation+Long+Expectation*Long|Item) + (1+Expectation+Long+Expectation*Long|Model), exp4 %>% filter((Expectation > 0 && Region == "CPLightVerb") || (Expectation < 0 &&Region == "MainVerb"))))
#summary(lmer(Surprisal ~ Expectation*Long + (1+Expectation+Long+Expectation*Long|Item) + (1+Expectation+Long+Expectation*Long|Model), exp4 %>% filter(Region == "PreLV")))


