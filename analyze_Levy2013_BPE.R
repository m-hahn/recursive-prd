
library(tidyr)
library(dplyr)
library(lme4)

models = read.csv("results/models_vanillaLSTM_BPE_russian.tsv", sep="\t")

data = data.frame()
for(model in models$Model) {
   data2 = read.csv(paste("output/Levy20131a_russian_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 data_ <- read.csv("../stimuli/Levy_etal_2013/expt1a-tokenized.tsv", sep="\t")
 data_$LineNumber = (1:nrow(data_))-1

 data_ = merge(data_, data, by=c("LineNumber"))

data_ = data_ %>% filter(!grepl("OOV", RegionLSTM))



agreement = paste(data_$Word, "</w>", sep="") == data_$RegionLSTM

mean(agreement)


data_ = data_ %>% mutate("RCType" = ifelse(Condition %in% c("C", "D"), "ORC", "SRC"))
data_ = data_ %>% mutate("Order" = ifelse(Condition %in% c("A", "C"), "default", "scrambled"))
data_ = data_ %>% mutate(ORC.C = (RCType == "ORC")-0.5)
data_ = data_ %>% mutate(scrambled.C = (Order == "scrambled")-0.5)

data_ = merge(data_, models, by=c("Model"), all=TRUE)

# Sanity-checking
unique(((data_ %>% group_by(Condition) %>% filter(Region == "V0"))) %>% select(Condition, Word))

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



data_ = data_ %>% mutate(Position_ = ifelse(!Region %in% c("V0", "N1"), Position,
					    ifelse(Position == 3 & Region == "V0", 3.75, 
  				            ifelse(Position == 3 & Region == "N1", 3,
				            ifelse(Position == 4 & Region == "V0", 3.75,
					    ifelse(Position == 4 & Region == "N1", 4.5, NA))))))

plot = ggplot(data_ %>% group_by(Region, AveragePerformance, Round, Model, Item, Condition, RCType, Order, Position_) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance, Position_, Region, RCType, Order) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Position_, y=Surprisal, group=Condition, color=RCType, linetype=Order))
plot = plot + geom_line()
plot = plot + geom_label(aes(label=Region))
plot = plot + facet_wrap(~AveragePerformance)
ggsave("figures/levy2013-fullplot.pdf")



# Relative Clause Verb
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data_ %>% filter(Region == "V0")))

#library(brms)
model = brm(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C+ORC.C * scrambled.C |Item) + (1+ORC.C+scrambled.C+ORC.C * scrambled.C |Model), data=data_ %>% filter(Region == "V0"))



data_ = data_ %>% mutate(Local = (Condition %in% c("A", "D"))) %>% mutate(Local.C = Local-mean(Local))



plot = ggplot(data_ %>% filter(Region %in% c("V0")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Item, Condition, AveragePerformance) %>% summarise(Surprisal=mean(Surprisal)) %>% transform(ItemF = as.character(Item)), aes(x=Condition, y=Surprisal, group=ItemF, color=ItemF))
plot = plot + geom_line()
plot = plot + facet_wrap(~AveragePerformance)



plot = ggplot(data_ %>% filter(Region %in% c("V0")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance, RCType, Order) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal, color=RCType, shape=Order))
plot = plot + geom_point()
plot = plot + facet_wrap(~AveragePerformance)
ggsave("figures/levy2013-relclverb.pdf")




# Relative Clause Noun
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data_ %>% filter(Region == "N1")))



plot = ggplot(data_ %>% filter(Region %in% c("N1")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Item, Condition, AveragePerformance) %>% summarise(Surprisal=mean(Surprisal)) %>% transform(ItemF = as.character(Item)), aes(x=Condition, y=Surprisal, group=ItemF, color=ItemF))
plot = plot + geom_line()
plot = plot + facet_wrap(~AveragePerformance)


plot = ggplot(data_ %>% filter(Region %in% c("N1")) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance, RCType, Order) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal, color=RCType, shape=Order))
plot = plot + geom_point()
plot = plot + facet_wrap(~AveragePerformance)
ggsave("figures/levy2013-relclnoun.pdf")




# End of RC
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data_ %>% filter(Region == "Prep")))

summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data_ %>% filter(Region == "U1")))

summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data_ %>% filter(Region == "U2")))

# Matrix Verb
summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data_ %>% filter(Position == 9)))


#plot = ggplot(data_ %>% filter(Position == 9) %>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance, RCType, Order) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal, color=RCType, shape=Order))
#plot = plot + geom_point()
#plot = plot + facet_wrap(~AveragePerformance)
#ggsave("figures/levy2013-relclnoun.pdf")





data__ = data_ %>% filter(Position < 9) %>% group_by(AveragePerformance, Round, Model, Item, RCType, Order, Condition, ORC.C, scrambled.C) %>% summarise(Surprisal = sum(Surprisal))



plot = ggplot(data__%>% group_by(AveragePerformance, Round, Model, Item, Condition, RCType, Order) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Condition, AveragePerformance, RCType, Order) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Condition, y=Surprisal, color=RCType, shape=Order))
plot = plot + geom_point()
plot = plot + facet_wrap(~AveragePerformance)
ggsave("figures/levy2013-aggregate.pdf")





summary(lmer(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C|Item) + (1+ORC.C+scrambled.C|Model), data=data__))

modelFull = brm(Surprisal ~ ORC.C + scrambled.C + ORC.C * scrambled.C + (1+ORC.C+scrambled.C+ ORC.C * scrambled.C|Item) + (1+ORC.C+scrambled.C+ ORC.C * scrambled.C|Model), data=data__)


plot = ggplot(data_ %>% filter(!HasParticle, Region %in% c("V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(data_ %>% filter(Region %in% c("A0", "V0")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 

library(lme4)

summary(lmer(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=data_ %>% filter(Region == "V0", !HasParticle)))

#library(brms)
#summary(brm(Surprisal ~ ORC.C + (1+ORC.C|Item) + (1+ORC.C|Model), data=data_ %>% filter(Region == "V0", !HasParticle)))



plot = ggplot(data_ %>% filter(Region %in% c("P0", "D2", "N2")) %>% group_by(Round, Model, Item, Condition, Length, Group) %>% summarise(Surprisal=sum(Surprisal)) %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 


plot = ggplot(data_ %>% filter(Region == "V1") %>% group_by(Length, Group) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Group, y=Surprisal, group=Length, color=Length, fill=Length)) + geom_bar(stat="identity", position=position_dodge(width=0.9)) 
ggsave(plot, file="figures/staub2016_vanilla_v1.pdf", width=18, height=3.5)




orc_data = data_ %>% filter(RCType == "ORC")

orc_data = orc_data %>% mutate(HasPP.C = HasPP-mean(HasPP))
orc_data = orc_data %>% mutate(HasParticle.C = HasParticle-mean(HasParticle))


summary(lmer(Surprisal ~ HasPP.C * HasParticle.C + (1+HasPP+HasParticle|Item) + (1+HasPP+HasParticle|Model), data=orc_data %>% filter(Region == "V1")))

#summary(lmer(Surprisal ~ Condition + (1+Condition|Item) + (1+Condition|Model), data=data_ %>% filter(Region == "v1")))

