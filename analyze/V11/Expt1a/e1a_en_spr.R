# From http://www.ling.uni-potsdam.de/~vasishth/code/VSLK_LCP.zip


## This file contains the non-public version of the data analyses for
##  Expt 1a presented in the paper:
#   Shravan Vasishth, Katja Suckow, Richard Lewis, and Sabine Kern.
#   Short-term forgetting in sentence comprehension:
#   Crosslinguistic evidence from head-final structures.
#   Submitted to Language and Cognitive Processes, 2007.

library(lme4)
library(car)
#library(coda)
library(Hmisc)

## Process data for analysis:
source("processdata.R",echo=T)


## orthogonal coding:
g <- ifelse(critdata$condition%in%c("a","b"),1,-1)
i <- ifelse(critdata$condition%in%c("a","c"),1,-1)
gxi <- ifelse(critdata$condition%in%c("a","d"),-1,1)

critdata$g <- g
critdata$i <- i
critdata$gxi <- gxi

## comparison 0 at np3:
data0 <- subset(critdata,region=="NP3")

summary(fm0 <- lmer(log(value)~ g + i + gxi + (1|subject)+(1|item),
                   data=data0))
summary(fm0 <- lmer(Surprisal~ g + i + gxi + (1|subject)+(1|item),
                   data=data0))


## comparison 1 at v3:
data1 <- subset(critdata,region=="V3")

summary(fm1 <- lmer(log(value)~ g + i + gxi + (1|subject)+(1|item),
                   data=subset(data1,value<2000)))
summary(fm1 <- lmer(Surprisal~ g + i + gxi + (1|subject)+(1|item),
                   data=subset(data1,value<2000)))


## comparison 2 V2 in gram and V1 in ungrammatical:

data2 <- subset(critdata,(region=="V2" & g==1) |
                                        (region=="V1" & g==-1))

## since word length differs in critical regions, add that as a covariate
data2$wl <- nchar(as.character(data2$word))

## compare word lengths
with(data2,tapply(wl,gram,mean))
with(data2,tapply(wl,gram,se))

summary(fm2 <- lmer(log(value)~ g+i+gxi+center(wl)+(1|subject)+(1|item),
                   data=data2))
summary(fm2 <- lmer(Surprisal~ g+i+gxi+center(wl)+(1|subject)+(1|item),
                   data=data2))


## comparison 3 V1:
data3 <- subset(critdata,(region=="V1"))


summary(fm3 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data3,value<2000)))

summary(fm3 <- lmer(value~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data3,value<2000)))

summary(fm3 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=data3))


summary(fm3 <- lmer(Surprisal~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data3,value<2000)))

summary(fm3 <- lmer(Surprisal~ g+i+gxi+(1|subject)+(1|item),
                   data=data3))

## comparison 4 post V1 region:
data4 <- subset(critdata,(region=="postV1"))

summary(fm4 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4,value<2000)))

summary(fm4 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4,value<2000)))


summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4,value<2000)))

summary(fm4 <- lmer(Surprisal~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4,value<2000)))


