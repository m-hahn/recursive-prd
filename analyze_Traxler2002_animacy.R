
library(tidyr)
library(dplyr)
library(lme4)


models = read.csv("results/models_vanillaLSTM_english.tsv", sep="\t")

models = c(
905843526,
655887140,
766978233,
502504068,
697117799,
860606598
)

data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Traxler2002_expt3_english_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv("../stimuli/traxler_etal_2002/expt3-tokenized.tsv", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

mean(as.character(raw.spr.data$Word) == as.character(raw.spr.data$RegionLSTM))

raw.spr.data = raw.spr.data %>% mutate(ORC.C=grepl("ORC", Condition)-0.5)
raw.spr.data = raw.spr.data %>% mutate(animate.C=grepl("mat_animate", Condition)-0.5)


#summary(lmer(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v0")))

#Matrix verb
summary(lmer(Surprisal ~ ORC.C*animate.C + (1+ORC.C+animate.C|Item) + (1+ORC.C+animate.C|Model), data=raw.spr.data %>% filter(Region == "v1")))
# Unexpected: animacy reduces effect of ORC (but the verbs are not balanced)
# But, effect size small

#library(brms)
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v0"), chains=1))
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "v1"), chains=1))


# Total surprisal on Relative Clause
relcl = raw.spr.data %>% filter(Region %in% c("d1", "n1", "v0")) %>% group_by(Model, Item, ORC.C, Round, animate.C) %>% summarise(Surprisal=sum(Surprisal))
summary(lmer(Surprisal ~ ORC.C*animate.C + (1+ORC.C+animate.C|Item) + (1+ORC.C+animate.C+ORC.C*animate.C|Model), data=relcl))
# Animate matrix noun makes ORC effect greater

# Total surprisal on embedded NP
np = raw.spr.data %>% filter(Region %in% c("d1", "n1")) %>% group_by(Model, Item, ORC.C, Round) %>% summarise(Surprisal=sum(Surprisal))
summary(lmer(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=np))

library(ggplot2)

plot = ggplot(raw.spr.data %>% group_by(Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Region, y=Surprisal, group=Condition, fill=Condition))
plot = plot + geom_col(position="dodge")


raw.spr.data = raw.spr.data %>% mutate(RegPosition = case_when(Region == "d0" ~ 1, Region == "n0" ~ 2, Region == "c" ~ 3, Region == "d1" & ORC.C > 0 ~ 4, Region == "n1" & ORC.C > 0 ~ 5, Region == "v0" & ORC.C > 0 ~ 6, Region == "v1" ~ 7, Region == "post" ~ 8, Region == "v0" & ORC.C<0 ~ 4, Region == "d1" & ORC.C<0 ~ 5, Region == "n1" & ORC.C<0 ~ 6))
raw.spr.data = raw.spr.data %>% mutate(RegWord = case_when(Region == "d0" ~ "the", Region == "n0" ~ "banker", Region == "c" ~ "that", Region == "d1" ~ "the", Region == "n1" ~ "lawyer", Region == "v0" ~ "irritated", Region == "v1" ~ "played", Region == "post" ~ "tennis every Sunday"))

plot = ggplot(raw.spr.data %>% group_by(RegWord, RegPosition, Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=RegPosition, y=Surprisal, group=Condition, color=Condition))
plot = plot + geom_line()
plot = plot + geom_text(aes(label=RegWord))



