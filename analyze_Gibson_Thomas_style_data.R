# PROBLEM the conditions are not matched for length

library(tidyr)
library(dplyr)
library(lme4)
#data = read.csv("output/GT_8066636", sep="\t")
data = read.csv("output/GT_wiki-english-nospaces-bptt-WHITESPACE-732605720", sep="\t")

data = data %>% filter(Type == "nested-sentence")
summary(data)
data = data %>% group_by(Item, Condition, Iteration) %>% summarise(Surprisal = sum(Surprisal))
data = data %>% mutate(O = grepl("O", Condition))
data = data %>% mutate(VP2 = grepl("VP2", Condition))
data = data %>% mutate(allVPs = grepl("allVPs", Condition))
data = data %>% filter(! allVPs)
summary(lmer(Surprisal ~ VP2 + (1 + VP2|Item), data=data %>% filter(O)))
summary(lmer(Surprisal ~ VP2 + (1 + VP2|Item), data=data %>% filter(!O)))
summary(lmer(Surprisal ~ VP2*O + (1 + VP2 + O + VP2*O|Item), data=data))

