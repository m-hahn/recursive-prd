###################################################
### chunk number 2: 
###################################################

library(reshape)
library(lme4)
#library(Hmisc)
library(tidyr)
library(dplyr)

models = c(
905843526 ,  
655887140 , 
766978233 ,
502504068 ,
697117799 ,
860606598 )

dataModels = data.frame()
for(model in models) {
  data2 = read.csv(paste("../../../output/V11_E1_ColorlessGreen_english_", model, sep=""), sep="\t")  %>% mutate(Model = model)
  dataModels = rbind(dataModels, data2)
}

data <- read.csv("../../../../stimuli/V11/colorlessGreen_English.txt", sep="\t")
data$LineNumber = (1:nrow(data))-1

mean(grepl("OOV", dataModels$RegionLSTM))

data = merge(data, dataModels, by=c("LineNumber")) %>% filter(!grepl("OOV", RegionLSTM))

mean(as.character(data$RegionLSTM) == as.character(data$Word))



data$g <- ifelse(data$Condition%in%c("g"),1,-1)
data$i <- 1
data$gxi <- 1



## comparison 0:
data0 <- subset(data,Region=="N3")

summary(fm0 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|Item),
                   data=data0))

## comparison 1:
data1 <- subset(data,Region=="V3")

summary(fm1 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|Item),
                   data=data1))


## comparison 2:

data2 <- subset(data,(Region=="V2" & Condition%in%c("g")) |
                                        (Region=="V1" & Condition%in%c("u")))

summary(fm2 <- lmer(Surprisal~ g+i+gxi+
                    #center(wl)+
                    (1|Model),#+(1|Item),
                    data=data2))



## comparison 3:
data3 <- subset(data,(Region=="V1"))

summary(fm3 <- lmer(Surprisal~ g+i+gxi +(1+g|Model)+(1+g|Item), data=data3)) # 


library(ggplot2)
plot = ggplot(data3 %>% group_by(g, Model) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line()


## comparison 4:
data4 <- subset(data,(Region=="D4"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|Item), data=data4)) # the opposite of structural forgetting (expected)

library(ggplot2)
plot = ggplot(data4 %>% group_by(g, Model) %>% summarise(Surprisal=mean(Surprisal)), aes(x=g, y=Surprisal, group=Model)) + geom_line()




## comparison 4:
data5 <- subset(data,(Region=="N4"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|Item), data=data5))




plot = ggplot(data %>% group_by(Model, Region, g) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Region, y=Surprisal, group=paste(Model, g))) + geom_line(aes(linetype=as.factor(g)))




