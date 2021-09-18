
library(tidyr)
library(dplyr)
library(lme4)

models = read.csv("results/models_vanillaLSTM_german.tsv", sep="\t")

data = data.frame()
for(model in models$Model) {
   data2 = read.csv(paste("output/Stone2018spr_german_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.csv("../stimuli/Stone_etal_2018/data_spr.txt", sep="\t")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

raw.spr.data = raw.spr.data %>% filter(!grepl("OOV", RegionLSTM))


mean(as.character(raw.spr.data$word) == as.character(raw.spr.data$RegionLSTM))

library(ggplot2)

critical = raw.spr.data %>% filter(region == "part")


ggplot(critical %>% group_by(Model, cond) %>% summarise(Surprisal = mean(Surprisal)), aes(x=cond, y=Surprisal, group = as.factor(Model), color=as.factor(Model))) + geom_line()

critical = critical %>% mutate(Large = (cond %in% c("c", "d")), Large.C=Large-0.5)
critical = critical %>% mutate(Long = (cond %in% c("b", "d")), Long.C=Long-0.5)
critical = critical %>% mutate(cloze.C = cloze - mean(cloze, na.rm=TRUE))
critical = critical %>% mutate(entropy.C = entropy - mean(entropy, na.rm=TRUE))


critical_ = critical %>% group_by(Large.C, Long.C, entropy.C, item, Model) %>% summarise(Surprisal=mean(Surprisal))

summary(lmer(Surprisal ~ Long.C*entropy.C + (1+Long.C+entropy.C|Model) + (1+Long.C+entropy.C|item), data=critical_))

summary(lmer(Surprisal ~ Long.C*entropy.C + (1+Long.C+entropy.C|Model) + (1+Long.C+entropy.C|item), data=critical_))


summary(lmer(Surprisal ~ Long.C*entropy.C + (1+Long.C+entropy.C+Long.C*entropy.C|Model) + (1+Long.C+entropy.C+Long.C*entropy.C|item), data=critical_))

summary(lmer(Surprisal ~ Large.C*Long.C + (1+Large.C+Long.C+Large.C*Long.C|Model) + (1+Large.C+Long.C+Large.C*Long.C|item), data=critical_))


####################

raw.spr.data = raw.spr.data %>% mutate("RCType" = ifelse(Condition %in% c("C", "D"), "ORC", "SRC"))
raw.spr.data = raw.spr.data %>% mutate("Order" = ifelse(Condition %in% c("A", "C"), "default", "scrambled"))
raw.spr.data = raw.spr.data %>% mutate(ORC.C = (RCType == "ORC")-0.5)
raw.spr.data = raw.spr.data %>% mutate(scrambled.C = (Order == "scrambled")-0.5)

raw.spr.data = merge(raw.spr.data, models, by=c("Model"), all=TRUE)

# Sanity-checking
unique(((raw.spr.data %>% group_by(Condition) %>% filter(Region == "V0"))) %>% select(Condition, Word))

#     conditionA = line
#     regionsA = ["N0", "Punct0", "Rel", "V0", "N1", "Prep", "U1", "U2", "Punct1", "Post"]
#
#     conditionB = [line[0], line[1], line[2], line[4], line[3]] + line[5:]
#     regionsB = ["N0", "Punct0", "Rel", "N1", "V0", "Prep", "U1", "U2", "Punct1", "Post"]
#
#     conditionC = [line[0], line[1], "которого", nominative, line[3]] + line[5:]
#     regionsC = ["N0", "Punct0", "Rel", "N1", "V0", "Prep", "U1", "U2", "Punct1", "Post"]
#
#     conditionD = [line[0], line[1], "которого", line[3], nominative] + line[5:]
#     regionsD = ["N0", "Punct0", "Rel", "V0", "N1", "Prep", "U1", "U2", "Punct1", "Post"]



library(ggplot2)



plot = ggplot(raw.spr.data %>% group_by(Region, AveragePerformance, Round, Model, Item, Condition, RCType, Order, Position) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance, Position, Region) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Position, y=Surprisal, group=Condition, color=Condition))
plot = plot + geom_line()
plot = plot + geom_label(aes(label=Region))
plot = plot + facet_wrap(~AveragePerformance)




# Relative Clause Verb
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=raw.spr.data %>% filter(Region == "V0")))



plot = ggplot(raw.spr.data %>% filter(Region %in% c("V0")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Item, Condition, AveragePerformance) %>% summarise(Surprisal=mean(Surprisal)) %>% transform(ItemF = as.character(Item)), aes(x=Condition, y=Surprisal, group=ItemF, color=ItemF))
plot = plot + geom_line()
plot = plot + facet_wrap(~AveragePerformance)



plot = ggplot(raw.spr.data %>% filter(Region %in% c("V0")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal))
plot = plot + geom_point()
plot = plot + facet_wrap(~AveragePerformance)


# Relative Clause Noun
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=raw.spr.data %>% filter(Region == "N1")))



plot = ggplot(raw.spr.data %>% filter(Region %in% c("N1")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Item, Condition, AveragePerformance) %>% summarise(Surprisal=mean(Surprisal)) %>% transform(ItemF = as.character(Item)), aes(x=Condition, y=Surprisal, group=ItemF, color=ItemF))
plot = plot + geom_line()
plot = plot + facet_wrap(~AveragePerformance)


plot = ggplot(raw.spr.data %>% filter(Region %in% c("N1")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal))
plot = plot + geom_point()
plot = plot + facet_wrap(~AveragePerformance)




# End of RC
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=raw.spr.data %>% filter(Region == "Prep")))

summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=raw.spr.data %>% filter(Region == "U1")))

summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=raw.spr.data %>% filter(Region == "U2")))

# Matrix Verb
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=raw.spr.data %>% filter(Position == 9)))




plot = ggplot(raw.spr.data %>% filter(!HasParticle, Region %in% c("V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(raw.spr.data %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 

library(lme4)

summary(lmer(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "V0", !HasParticle)))

#library(brms)
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=raw.spr.data %>% filter(Region == "V0", !HasParticle)))



plot = ggplot(raw.spr.data %>% filter(Region %in% c("P0", "D2", "N2")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(raw.spr.data %>% filter(Region == "V1") %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
ggsave(plot, file="figures/staub2016_vanilla_v1.pdf", width=18, height=3.5)




orc_data = raw.spr.data %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))


summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=raw.spr.data %>% filter(Region == "v1")))

