
library(tidyr)
library(dplyr)
library(lme4)

models = c(
561380584 , 590722918 , 682614531 , 837532417 , 847004279 , 864344911  
)

data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Safavi2016_farsi_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 surpdata <- read.csv("stimuli/safavi_etal_2016_persian/items_merged_tokenized.txt", sep="\t")
 surpdata$LineNumber = (1:nrow(surpdata))-1

 surpdata = merge(surpdata, data, by=c("LineNumber"))

mean(as.character(surpdata$Word) == as.character(surpdata$RegionLSTM))



library(lme4)


surpdata = surpdata %>% mutate(long = ifelse(Condition %in% c(1,3), -0.5, 0.5))
surpdata = surpdata %>% mutate(light_verb = ifelse(Condition %in% c(1,2), 0.5, -0.5))
surpdata = surpdata %>% mutate(expt2 = (Part-1.5)/0.5)
surpdata = surpdata %>% mutate(Item = Item + 40*(Part-1))

summary(lmer(Surprisal ~  long*light_verb + (1+long+light_verb+long*light_verb|Item) + (1+long+light_verb+long*light_verb|Model), data=surpdata %>% filter(Region == "verb", Part==1)))
# - locality effect
# - locality - predictability interaction, but in the unexpected direction

summary(lmer(Surprisal ~  long*light_verb + (1+long+light_verb+long*light_verb|Item) + (1+long+light_verb+long*light_verb|Model), data=surpdata %>% filter(Region == "verb", Part==2)))
# - no locality effect
# - interaction, in the expected direction


summary(lmer(Surprisal ~  expt1*long*light_verb + (1+long+light_verb+long*light_verb|Item) + (1+long+light_verb+long*light_verb|Model), data=surpdata %>% filter(Region == "verb")))

library(ggplot2)
plot = ggplot(surpdata %>% filter(Region == "verb") %>% group_by(Model, long, light_verb, Part) %>% summarise(Surprisal=mean(Surprisal)), aes(x=long, y=Surprisal, group=Model)) + geom_line() + facet_wrap(~Part+light_verb)

