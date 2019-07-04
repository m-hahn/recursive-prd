# From http://www.ling.uni-potsdam.de/~vasishth/code/VSLK_LCP.zip


## This file contains the non-public version of the data analyses for
##  Expt 1a presented in the paper:
#   Shravan Vasishth, Katja Suckow, Richard Lewis, and Sabine Kern.
#   Short-term forgetting in sentence comprehension:
#   Crosslinguistic evidence from head-final structures.
#   Submitted to Language and Cognitive Processes, 2007.

## load data and prepare:

## reading time data analysis:

## preliminary data processing:
data <- read.table("~/scr/recursive-prd/VSLK_LCP/E1a_EN_SPR/data/e1a_en_spr_data.txt")
colnames(data) <- c("subject","expt","item","condition","position","word","rt")
data$LineNumber = (1:nrow(data))-1
dataModel = read.csv("../../../output/V11_Expt1_7032387", sep="\t")
library(tidyr)
library(dplyr)
dataModel = dataModel %>% filter(!grepl("OOV", RegionLSTM))
data = merge(data, dataModel, by=c("LineNumber"))


## make every position start with 1 instead of 0
data$position <- as.numeric(data$position)+1


library(reshape)

## computations cached:
d.rs <- melt(data, id=colnames(data),
                measure="rt", variable_name="times",
                na.rm=FALSE)


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


pos05data <- subset(d.rs,roi==5)

d.rs.NP3 <- melt(pos05data,
                 id=colnames(pos05data)[c(1,2,3,4,5,6, 10)],
                measure=c("rt"), variable_name="times",
                na.rm=FALSE)

sum <- rep(NA,dim(d.rs.NP3)[1])
detrt <- 0

for(i in 1:dim(d.rs.NP3)[1]){
  if(i%%2==0){## even row                               
     sum[i] <- detrt+d.rs.NP3[i,]$value}
  else {detrt <- d.rs.NP3[i,]$value} ## odd row
}

d.rs.NP3$sumrt <- sum
d.rs.NP3 <- subset(d.rs.NP3,pos05data$word!="the")
d.rs.NP3$value <- d.rs.NP3$sumrt
d.rs.NP3 <- d.rs.NP3[,1:9]

d.rs.NP3$gram <- ifelse(d.rs.NP3$condition%in%c("a","b"),"gram","ungram")
d.rs.NP3$int <- ifelse(d.rs.NP3$condition%in%c("a","c"),"hi","lo")


## V3:

pos09data <- subset(d.rs,position==9) ## had
pos10data <- subset(d.rs,position==10) 

pos10data$pos09rt <- pos09data$rt
pos10data$sumrt <- pos10data$value+pos10data$pos09rt
pos10data$meanrt <- (pos10data$value+pos10data$pos09rt)/2

hist(pos10data$sumrt)

d.rs.V3 <- melt(pos10data, id=colnames(pos10data)[c(1,2,3,4,5,6, 10)],
                measure=c("sumrt","meanrt"), variable_name="times",
                na.rm=FALSE)

## sanity check
unique(d.rs.V3$word)

## code up factors:
d.rs.V3$gram <- ifelse(d.rs.V3$condition%in%c("a","b"),
                       "gram","ungram")
d.rs.V3$int <- ifelse(d.rs.V3$condition%in%c("a","c"),
                      "hi","lo")
summary(d.rs.V3)
#hist(subset(d.rs.V3,times=="sumrt" & value<2000)$value)



## V2, relevant only for conditions a and b

data.ab <- subset(data,condition%in%c("a","b"))
data.cd <- subset(data,condition%in%c("c","d"))

pos15data <- subset(data.ab,position==15) ## V2 in a,b
auxrts <- subset(data.ab,position==14)$rt ## auxiliary rts
pos15data$sumrt <- pos15data$rt+auxrts
pos15data$meanrt <- (pos15data$rt+auxrts)/2

d.rs.V2ab <- melt(pos15data, id=colnames(pos15data)[c(1:6, 10)],
                measure=c("sumrt","meanrt"), variable_name="times",
                na.rm=TRUE)

d.rs.V2ab$gram <- ifelse(d.rs.V2ab$condition%in%c("a","b"),
                         "gram","ungram")

d.rs.V2ab$int <- ifelse(d.rs.V2ab$condition%in%c("a"),"hi",
                        ifelse(d.rs.V2ab$condition%in%c("b"),"lo",
                               NA))

## V1:

pos17data <- subset(data.ab,position==17)
pos14data <- subset(data.cd,position==14)
pos1714data <- rbind(pos17data,pos14data)

d.rs.V1 <- melt(pos1714data, id=colnames(pos1714data)[c(1:6,10)],
                measure="rt", variable_name="times",
                na.rm=TRUE)

d.rs.V1$gram <- ifelse(d.rs.V1$condition%in%c("a","b"),
                       "gram","ungram")
d.rs.V1$int <- ifelse(d.rs.V1$condition%in%c("a","c"),
                       "hi","lo")



## Post V1:

pos18data <- subset(data.ab,position==18)
pos15data <- subset(data.cd,position==15)
pos1815data <- rbind(pos18data,pos15data)

d.rs.postV1 <- melt(pos1815data, id=colnames(pos1815data)[c(1:6, 10)],
                measure="rt", variable_name="times",
                na.rm=TRUE)

d.rs.postV1$gram <- factor(ifelse(d.rs.postV1$condition%in%c("a","b"),
                                  "gram","ungram"))

d.rs.postV1$int <- factor(ifelse(d.rs.postV1$condition%in%c("a","c"),
                                  "hi","lo"))



d.rs.V3 <- subset(d.rs.V3,times=="sumrt")

d.rs.V3$gram <- factor(d.rs.V3$gram) 
d.rs.V3$times <- factor(d.rs.V3$times) 

d.rs.NP3 <- data.frame(region="NP3",d.rs.NP3)
d.rs.V3 <- data.frame(region="V3",d.rs.V3)
d.rs.V2ab <- data.frame(region="V2",subset(d.rs.V2ab,
                          times=="sumrt"))
d.rs.V1 <- data.frame(region="V1",d.rs.V1)
d.rs.postV1 <- data.frame(region="postV1",d.rs.postV1)

critdata <- rbind(d.rs.NP3,
                  d.rs.V3,
                  d.rs.V2ab,
                  d.rs.V1,
                  d.rs.postV1)

summary(critdata)

critdata$times <- factor("rt")

## end preliminaries
