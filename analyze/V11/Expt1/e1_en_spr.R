## This file contains the non-final version of the data analyses for
##  Expt 1 presented in the paper:
#   Shravan Vasishth, Katja Suckow, Richard Lewis, and Sabine Kern.
#   Short-term forgetting in sentence comprehension:
#   Crosslinguistic evidence from head-final structures.
#   Submitted to Language and Cognitive Processes, 2007.

library(lme4)

#library(Hmisc)
library(xtable)
library(MASS)

## interpretation:

## coefficient for gram:

## Process data for analysis:

## load data and prepare:

## reading time data analysis:
library(tidyr)
library(dplyr)
## preliminary data processing:
data <- read.table("../../../../../recursive-prd/VSLK_LCP/E1_EN_SPR/data/e1_en_spr_data.txt")
colnames(data) <- c("subject","expt","item","condition","position","word","rt")
data$LineNumber = (1:nrow(data))-1

models = c(
905843526 ,  
655887140 , 
766978233 ,
502504068 ,
697117799 ,
860606598 )

datModel = data.frame()
for(model in models) {
   datModel2 = read.csv(paste("../../../output/V11_E1_english_", model, sep=""), sep="\t") %>% mutate(Model = model)
   datModel = rbind(datModel, datModel2)
}

data = merge(data, datModel, by=c("LineNumber"))

data = data %>% filter(!grepl("OOV", RegionLSTM))

data <- subset(data,expt=="gug")
data$expt <- factor(data$expt)

## make every position start with 1 instead of 0
data$position <- as.numeric(data$position)+1


data$gram <- ifelse(data$condition%in%c("a","b"),"gram","ungram")
data$int <- ifelse(data$condition%in%c("a","c"),"hi","lo")


library(reshape)

## computations cached:
d.rs <- melt(data, id=colnames(data),
                measure="rt", variable_name="times",
                na.rm=TRUE)

unique(subset(d.rs,position==1)$word)


## recode regions of interest:

## The painter who the film that the friend liked admired the poet.  
##  1    2      3   4   5   6     7   8     9     10      11  12
##    1         2     3     4       5 

## recode the regions of interest:
d.rs$roi <- ifelse(d.rs$position==2,1, # NP1
               ifelse(d.rs$position==3,2, # who
                 ifelse(d.rs$position%in%c(4,5),3, #NP2
                            ifelse(d.rs$position==6,4, # that
                                   ifelse(d.rs$position%in%c(7,8),5, # NP3
                                                 d.rs$position)))))

## NP3:
pos05data <- subset(d.rs,roi==5)

d.rs.NP3 <- melt(pos05data, id=c("subject" ,  "expt"  ,    "item"   ,   "condition" ,"position" , "word", "Surprisal", "Model"),
                measure=c("rt"), variable_name="times",
                na.rm=FALSE)

sum <- rep(NA,nrow(d.rs.NP3))
detrt <- 0

for(i in 1:nrow(d.rs.NP3)){
  if(i%%2==0){## even row                               
     sum[i] <- detrt+d.rs.NP3[i,]$value}
  else {detrt <- d.rs.NP3[i,]$value} ## odd row
}

d.rs.NP3$sumrt <- sum
d.rs.NP3 <- subset(d.rs.NP3,word!="the")
d.rs.NP3$value <- d.rs.NP3$sumrt
d.rs.NP3$sumrt <- NULL

d.rs.NP3$gram <- ifelse(d.rs.NP3$condition%in%c("a","b"),"gram","ungram")
d.rs.NP3$int <- ifelse(d.rs.NP3$condition%in%c("a","c"),"hi","lo")


## V3:
pos09data <- subset(d.rs,position==9)

d.rs.V3 <- melt(pos09data, id=c("subject" ,  "expt"  ,    "item" ,     "condition" ,"position" , "word", "Surprisal", "Model"),
                measure=c("rt"), variable_name="times",
                na.rm=FALSE)

## sanity check
unique(d.rs.V3$word)

## code up factors:
d.rs.V3$gram <- ifelse(d.rs.V3$condition%in%c("a","b"),"gram","ungram")
d.rs.V3$int <- ifelse(d.rs.V3$condition%in%c("a","c"),"hi","lo")


## V2, relevant only for conditions a and b

data.ab <- subset(data,condition%in%c("a","b"))
data.cd <- subset(data,condition%in%c("c","d"))

pos10data <- subset(data.ab,position==10) ## V2 in a,b

d.rs.V2ab <- melt(pos10data, id=c("subject" ,  "expt"   ,   "item"   ,   "condition", "position" , "word", "Surprisal", "Model"),
                measure=c("rt"), variable_name="times",
                na.rm=TRUE)

d.rs.V2ab$gram <- ifelse(d.rs.V2ab$condition%in%c("a","b"),"gram","ungram")

d.rs.V2ab$int <- ifelse(d.rs.V2ab$condition%in%c("a"),"hi",
                        ifelse(d.rs.V2ab$condition%in%c("b"),"lo",NA))


## V1:

pos11data <- subset(data.ab,position==11)
pos10data <- subset(data.cd,position==10)
pos1011data <- rbind(pos10data,pos11data)

d.rs.V1 <- melt(pos1011data, id=c("subject" ,  "expt"  ,    "item"   ,   "condition", "position",  "word", "Surprisal", "Model"),
                measure="rt", variable_name="times",
                na.rm=TRUE)

d.rs.V1$gram <- ifelse(d.rs.V1$condition%in%c("a","b"),"gram","ungram")
d.rs.V1$int <- ifelse(d.rs.V1$condition%in%c("a","c"),"hi","lo")


## Post V1:

pos12data <- subset(data.ab,position==12)
pos11data <- subset(data.cd,position==11)
pos1112data <- rbind(pos12data,pos11data)

d.rs.postV1 <- melt(pos1112data, id=c("subject"  , "expt"    ,  "item"  ,    "condition", "position" , "word", "Surprisal", "Model"),
                measure="rt", variable_name="times",
                na.rm=TRUE)

d.rs.postV1$gram <- factor(ifelse(d.rs.postV1$condition%in%c("a","b"),"gram","ungram"))
d.rs.postV1$int <- factor(ifelse(d.rs.postV1$condition%in%c("a","c"),"hi","lo"))




d.rs.V3$gram <- factor(d.rs.V3$gram) 
d.rs.V3$times <- factor(d.rs.V3$times) 

d.rs.NP3 <- data.frame(region="NP3",d.rs.NP3)
d.rs.V3 <- data.frame(region="V3",d.rs.V3)
d.rs.V2ab <- data.frame(region="V2",d.rs.V2ab)
d.rs.V1 <- data.frame(region="V1",d.rs.V1)
d.rs.postV1 <- data.frame(region="postV1",d.rs.postV1)

critdata <- rbind(d.rs.NP3,d.rs.V3,d.rs.V2ab,d.rs.V1,d.rs.postV1)

critdata$times <- factor("rt")

## end preliminaries
#save(verbrts,file="verbrts.Rda")
#save(critdata,file="critdata.Rda")


summary(critdata)

## reading time per character:
num.char<-nchar(as.character(critdata$word))
rtperchar<-critdata$value/num.char


critdata <-subset(critdata,value<2000)
  
gram <- ifelse(critdata$condition%in%c("a","b"),-1,1)
int <- ifelse(critdata$condition%in%c("a","c"),-1,1)

critdata$gram <- gram
critdata$int <- int

## remove item 11, there was a mistake in it:
critdata <- subset(critdata,item!=11)

## main graphic with matplot




## orthogonal coding:
g <- ifelse(critdata$condition%in%c("a","b"),1,-1)
i <- ifelse(critdata$condition%in%c("a","c"),1,-1)
gxi <- ifelse(critdata$condition%in%c("a","d"),1,-1)

critdata$g <- g
critdata$i <- i
critdata$gxi <- gxi

## comparison 0 at np3:
data0 <- subset(critdata,region=="NP3")

summary(fm0 <- lmer(log(value)~ g + i + gxi + (1|subject)+(1|item),
                   data=data0))


## comparison 1 at v3:
data1 <- subset(critdata,region=="V3")

summary(fm1 <- lmer(log(value)~ g + i + gxi + (1|subject)+(1|item),
                   data=data1))



## comparison 2 V2 in gram and V1 in ungrammatical:

data2 <- subset(critdata,(region=="V2" & g==1) |
                                        (region=="V1" & g==-1))

## since word length differs in critical regions, add that as a covariate
data2$wl <- nchar(as.character(data2$word))

## compare word lengths
with(data2,tapply(wl,gram,mean))

data2 = data2 %>% mutate(wl_c = wl - mean(wl))
summary(fm2 <- lmer(log(value)~ g+i+gxi+wl_c+(1|subject)+(1|item),
                   data=data2))



## comparison 3 V1:
data3 <- subset(critdata,(region=="V1"))

summary(fm3 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=data3))

summary(fm3 <- lmer(value~ g+i+gxi+(1|subject)+(1|item),
                   data=data3))

summary(fm3 <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|item), # no visible difference here
                   data=data3))

## comparison 4 post V1 region (this is the crucial critical region for the grammaticality difference):
data4 <- subset(critdata,(region=="postV1"))

summary(fm4 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4)))

summary(fm4 <- lmer(value~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4)))

## this model is better
summary(fm4a <- lmer(log(value)~ g+i+gxi+(1+g|subject)+(1|item),
                   data=subset(data4)))

summary(fm4a <- lmer(Surprisal~ g+i+gxi+(1+g|Model)+(1+g|item), data=subset(data4)))
# What is i?
#g           -0.193967   0.037362  -5.192    # the opposite of structural forgetting
#i           -0.044981   0.009165  -4.908    # show interference facilitation
#gxi          0.017126   0.009167   1.868

#In Experiments 14 presented here, another factor was included, but this was orthogonal tothe present issue. This other factor was interference; we manipulated the similarity of the secondNP with respect to the first and third NPs. Since the results of that manipulation do not concernthis study,  we  omit  discussion  of this  factor in  the  paper.  The  items  presented in  Appendix 1show all four conditions, and the experimental data (which will be made available online), willcontain a full discussion of the interference results and their interpretation


library(ggplot2)
plot = ggplot(data %>% group_by(Model, position, gram) %>% summarise(Surprisal = mean(Surprisal)), aes(x=position, y=Surprisal, group=paste(Model, gram), color=gram)) + geom_line()
plot = ggplot(data %>% group_by(position, gram) %>% summarise(Surprisal = mean(Surprisal)), aes(x=position, y=Surprisal, group=gram, color=gram)) + geom_line()

