###################################################
### chunk number 2: 
###################################################

library(reshape)
library(lme4)
library(Hmisc)
library(tidyr)
library(dplyr)

models = c(
710899603  , 
553555034   ,
657350374   ,
887261094   ,
459934402   
)

dataModels = data.frame()

for(model in models) {
  data2 = read.csv(paste("../../../output/V11_E3_german_", model, sep=""), sep="\t")  %>% mutate(Model = model)
  dataModels = rbind(dataModels, data2)
}

data <- read.table("../../../../../recursive-prd/VSLK_LCP/E3_DE_SPR/data/e3desprdata.txt")
colnames(data) <- c("subj","expt","item","condition","position","word","RT","similarity","grammaticality")
data$LineNumber = (1:nrow(data))-1

data = merge(data, dataModels, by=c("LineNumber")) %>% filter(!grepl("OOV", RegionLSTM))

NP3.data <- subset(data,position==5)
V3.data <- subset(data,position==6)
V2.data <- subset(data,(condition%in%c("a","b") & position==7))
V1.data <- subset(data,(condition%in%c("a","b") & position==8) | (condition%in%c("c","d") & position==7))
postV1.data <- subset(data,(condition%in%c("a","b") & position==9) |
                      (condition%in%c("c","d") & position==8))

d.NP3.rs <- melt(NP3.data, id=c("subj", "condition","item","word", "Model","Surprisal"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.NP3.rs <- subset(d.NP3.rs,value>0)

d.NP3.rs$gram <- ifelse(d.NP3.rs$condition%in%c("a","b"),"gram","ungram")
d.NP3.rs$int <- ifelse(d.NP3.rs$condition%in%c("a","c"),"hi","lo")

d.V3.rs <- melt(V3.data, id=c("subj", "condition","item","word", "Model","Surprisal"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.V3.rs <- subset(d.V3.rs,value>0)

d.V3.rs$gram <- ifelse(d.V3.rs$condition%in%c("a","b"),"gram","ungram")
d.V3.rs$int <- ifelse(d.V3.rs$condition%in%c("a","c"),"hi","lo")


d.V2.rs <- melt(V2.data, id=c("subj", "condition","item","word", "Model","Surprisal"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.V2.rs <- subset(d.V2.rs,value>0)

d.V2.rs$gram <- ifelse(d.V2.rs$condition%in%c("a","b"),"gram",NA)
d.V2.rs$int <- ifelse(d.V2.rs$condition%in%c("a"),"hi","lo")

d.V1.rs <- melt(V1.data, id=c("subj", "condition","item","word", "Model","Surprisal"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.V1.rs <- subset(d.V1.rs,value>0)

d.V1.rs$gram <- ifelse(d.V1.rs$condition%in%c("a","b"),"gram","ungram")
d.V1.rs$int <- ifelse(d.V1.rs$condition%in%c("a","c"),"hi","lo")

d.postV1.rs <- melt(postV1.data, id=c("subj", "condition","item","word", "Model","Surprisal"),
              measure="RT", variable_name="times", na.rm=TRUE)
d.postV1.rs <- subset(d.postV1.rs,value>0)

d.postV1.rs$gram <- ifelse(d.postV1.rs$condition%in%c("a","b"),"gram","ungram")
d.postV1.rs$int <- ifelse(d.postV1.rs$condition%in%c("a","c"),"hi","lo")


d.NP3.rs <- data.frame(region="NP3",d.NP3.rs)
d.V3.rs <- data.frame(region="V3",d.V3.rs)
d.V2.rs <- data.frame(region="V2",d.V2.rs)
d.V1.rs <- data.frame(region="V1",d.V1.rs)
d.postV1.rs <- data.frame(region="postV1",d.postV1.rs)

d.rs <- rbind(d.NP3.rs,d.V3.rs,d.V2.rs,d.V1.rs,d.postV1.rs)

head(d.rs)
summary(d.rs)

#centered length
#d.V.rs$length <- center(nchar(as.character(d.V.rs$word)))

critdata <- d.rs

g <- ifelse(critdata$condition%in%c("a","b"),1,-1)
i <- ifelse(critdata$condition%in%c("a","c"),1,-1)
gxi <- ifelse(critdata$condition%in%c("a","d"),1,-1)

critdata$g <- g
critdata$i <- i
critdata$gxi <- gxi


summary(critdata)

## comparison 0:
data0 <- subset(critdata,region=="NP3")

summary(fm0 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|item),
                   data=data0))

## comparison 1:
data1 <- subset(critdata,region=="V3")

summary(fm1 <- lmer(Surprisal~ g+i+gxi+ (1|Model)+(1|item),
                   data=data1))


## comparison 2:

data2 <- subset(critdata,(region=="V2" & condition%in%c("a","b")) |
                                        (region=="V1" & condition%in%c("c","d")))

summary(fm2 <- lmer(Surprisal~ g+i+gxi+
                    #center(wl)+
                    (1|Model),#+(1|item),
                    data=data2))



## comparison 3:
data3 <- subset(critdata,(region=="V1"))

summary(fm3 <- lmer(Surprisal~ g+i+gxi +(1|Model)+(1|item),
                   data=subset(data3,value<2000)))

## comparison 4:
data4 <- subset(critdata,(region=="postV1"))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1|Model)+(1|item),
                   data=data4))




