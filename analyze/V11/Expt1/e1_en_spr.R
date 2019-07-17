## This file contains the non-final version of the data analyses for
##  Expt 1 presented in the paper:
#   Shravan Vasishth, Katja Suckow, Richard Lewis, and Sabine Kern.
#   Short-term forgetting in sentence comprehension:
#   Crosslinguistic evidence from head-final structures.
#   Submitted to Language and Cognitive Processes, 2007.

library(lme4)
library(car)

library(Hmisc)
library(xtable)
library(MASS)

## interpretation:

## coefficient for gram:

## Process data for analysis:

## load data and prepare:

## reading time data analysis:

## preliminary data processing:
data <- read.table("~/scr/recursive-prd/VSLK_LCP/E1_EN_SPR/data/e1_en_spr_data.txt")
colnames(data) <- c("subject","expt","item","condition","position","word","rt")

data <- subset(data,expt=="gug")
data$expt <- factor(data$expt)

## make every position start with 1 instead of 0
data$position <- as.numeric(data$position)+1

## don't need this
#dataq <- read.table("engugspr1q.txt")
#colnames(dataq) <- c("subj","expt","item","condition","dummy","response","correct","RT")

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

d.rs.NP3 <- melt(pos05data, id=colnames(pos05data)[c(1,2,3,4,5,6)],
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
d.rs.NP3 <- subset(d.rs.NP3,word!="the")
d.rs.NP3$value <- d.rs.NP3$sumrt
d.rs.NP3 <- d.rs.NP3[,1:8]

d.rs.NP3$gram <- ifelse(d.rs.NP3$condition%in%c("a","b"),"gram","ungram")
d.rs.NP3$int <- ifelse(d.rs.NP3$condition%in%c("a","c"),"hi","lo")

## means and CIs for plotting
d.cast.NP3.rt.i   <- cast(d.rs.NP3, int ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

## V3:
pos09data <- subset(d.rs,position==9)

d.rs.V3 <- melt(pos09data, id=colnames(pos09data)[c(1,2,3,4,5,6)],
                measure=c("rt"), variable_name="times",
                na.rm=FALSE)

## sanity check
unique(d.rs.V3$word)

## code up factors:
d.rs.V3$gram <- ifelse(d.rs.V3$condition%in%c("a","b"),"gram","ungram")
d.rs.V3$int <- ifelse(d.rs.V3$condition%in%c("a","c"),"hi","lo")

## means and CIs for plotting
d.cast.V3.rt.g   <- cast(d.rs.V3, gram ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

d.cast.V3.rt.i   <- cast(d.rs.V3, int ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

## V2, relevant only for conditions a and b

data.ab <- subset(data,condition%in%c("a","b"))
data.cd <- subset(data,condition%in%c("c","d"))

pos10data <- subset(data.ab,position==10) ## V2 in a,b

d.rs.V2ab <- melt(pos10data, id=colnames(pos10data)[c(1:6)],
                measure=c("rt"), variable_name="times",
                na.rm=TRUE)

d.rs.V2ab$gram <- ifelse(d.rs.V2ab$condition%in%c("a","b"),"gram","ungram")

d.rs.V2ab$int <- ifelse(d.rs.V2ab$condition%in%c("a"),"hi",
                        ifelse(d.rs.V2ab$condition%in%c("b"),"lo",NA))


d.cast.V2ab.g   <- cast(d.rs.V2ab, gram ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))),
                        subset=times=="rt")

d.cast.V2ab.i   <- cast(d.rs.V2ab, int ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))),
                        subset=times=="rt")

## V1:

pos11data <- subset(data.ab,position==11)
pos10data <- subset(data.cd,position==10)
pos1011data <- rbind(pos10data,pos11data)

d.rs.V1 <- melt(pos1011data, id=colnames(pos1011data)[1:6],
                measure="rt", variable_name="times",
                na.rm=TRUE)

d.rs.V1$gram <- ifelse(d.rs.V1$condition%in%c("a","b"),"gram","ungram")
d.rs.V1$int <- ifelse(d.rs.V1$condition%in%c("a","c"),"hi","lo")

d.cast.V1.g  <- cast(d.rs.V1, gram ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

d.cast.V1.i  <- cast(d.rs.V1, int ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

## Post V1:

pos12data <- subset(data.ab,position==12)
pos11data <- subset(data.cd,position==11)
pos1112data <- rbind(pos12data,pos11data)

d.rs.postV1 <- melt(pos1112data, id=colnames(pos1112data)[1:6],
                measure="rt", variable_name="times",
                na.rm=TRUE)

d.rs.postV1$gram <- factor(ifelse(d.rs.postV1$condition%in%c("a","b"),"gram","ungram"))
d.rs.postV1$int <- factor(ifelse(d.rs.postV1$condition%in%c("a","c"),"hi","lo"))


d.cast.postV1.g   <- cast(d.rs.postV1, gram ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

d.cast.postV1.i   <- cast(d.rs.postV1, int ~ .,
                 function(x) c(M=round(mean(x)),
                               CI=round(ci(x))), subset=times=="rt")

verbrts.g <- rbind(d.cast.V3.rt.g,
                 d.cast.V2ab.g,
                 c("ungram","NA","NA","NA"),
                 d.cast.V1.g,
                 d.cast.postV1.g)
verbrts.g$verb <- rep(c("V3","V2","V1","Post-V1"),each=2)
verbrts.g$region <- rep(1:4,each=2)
names(verbrts.g$region) <- "Region"
levels(verbrts.g$region) <- verbrts.g$verb
verbrts.g$CI.upper <- as.numeric(verbrts.g$CI.upper)
verbrts.g$CI.lower <- as.numeric(verbrts.g$CI.lower)
verbrts.g$M <- as.numeric(verbrts.g$M)

verbrts.i <- rbind(d.cast.NP3.rt.i,
                   d.cast.V3.rt.i,
                   d.cast.V2ab.i,
                   d.cast.V1.i,
                   d.cast.postV1.i)

verbrts.i$verb <- rep(c("NP3","V3","V2","V1","Post-V1"),each=2)
verbrts.i$region <- rep(1:5,each=2)
names(verbrts.i$region) <- "Region"
levels(verbrts.i$region) <- verbrts.i$verb
verbrts.i$CI.upper <- as.numeric(verbrts.i$CI.upper)
verbrts.i$CI.lower <- as.numeric(verbrts.i$CI.lower)
verbrts.i$M <- as.numeric(verbrts.i$M)

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

load("critdata.Rda")

summary(critdata)

## reading time per character:
num.char<-nchar(as.character(critdata$word))
rtperchar<-critdata$value/num.char

plot(critdata$value~num.char)
abline(lm(critdata$value~num.char))
lines(lm(critdata$value~num.char))

plot(critdata$value~scale(num.char,scale=FALSE))
abline(lm(critdata$value~scale(num.char,scale=FALSE)))


plot(log(critdata$value)~num.char)
abline(lm(log(critdata$value)~num.char))

plot(rtperchar~num.char)
abline(lm(rtperchar~num.char))

plot(log(rtperchar)~num.char)
abline(lm(log(rtperchar)~num.char))



boxplot(log(critdata$value))
boxplot(log(subset(critdata,value<2000)$value))

dim(critdata)
dim(subset(critdata,value<2000))


critdata <-subset(critdata,value<2000)
  
gram <- ifelse(critdata$condition%in%c("a","b"),-1,1)
int <- ifelse(critdata$condition%in%c("a","c"),-1,1)

critdata$gram <- gram
critdata$int <- int

## remove item 11, there was a mistake in it:
critdata <- subset(critdata,item!=11)

## main graphic with matplot
gram.means <- subset(verbrts.g,gram=="gram")
ungram.means <- subset(verbrts.g,gram=="ungram")

graycol <- gray(.5)

filename <- "e1ensprplotgram.ps"
createPS(filename)

matplot(gram.means$M,
        ylim=c(min(verbrts.g$CI.lower,na.rm=T),
               max(verbrts.g$CI.upper,na.rm=T)),
        main="Experiment 1 (English SPR)",
        type="b",
        pch=16,
        lty=1,
        xaxt="n",
        cex.main=1.8,
        lwd=5,
        ylab="",
        cex.axis=1.8)

lines(ungram.means$M,lwd=3,lty=2,col=graycol)
points(ungram.means$M,pch=19,cex=1.5,col=graycol)

arrows(seq(from=1,4,by=1),gram.means$CI.lower,
       seq(from=1,4,by=1),gram.means$CI.upper,
       lwd =  2,col="black",angle=90,code=3,length=.1)

arrows(seq(from=1,4,by=1),ungram.means$CI.lower,
       seq(from=1,4,by=1),ungram.means$CI.upper,
       lwd =  2,col=graycol,angle=90,code=3,length=.1)

mtext(text=c("V3","V2","V1","Post-V1"),side=1,line=.75,at=c(1,2,3,3.9),cex=2)
mtext(text="Reading time [ms]",side=2,line=2.5,cex=2)
mtext(text="Region",side=1,line=2.5,cex=2)

legend(2,1200,lty=c(1,2),col=c("black",graycol),pch=c(16,19),
       legend=c("grammatical","ungrammatical"),cex=2,lwd=3)

dev.off()

dest <- "~/Dropbox/Papers/IntML/Figures"
  
destination <- dest
system(paste("cp ",filename,destination,sep=" "))

## interference plot
hi.means <- subset(verbrts.i,int=="hi")
lo.means <- subset(verbrts.i,int=="lo")

filename <- "e1ensprplotint.ps"
createPS(filename)

matplot(hi.means$M,
        ylim=c(min(verbrts.i$CI.lower,na.rm=T),
               max(verbrts.i$CI.upper,na.rm=T)),
        main="Experiment 1 (English SPR): interference",
        type="b",
        pch=16,
        lty=1,
        xaxt="n",
        cex.main=1.8,
        lwd=5,
        ylab="",
        cex.axis=1.8)

lines(lo.means$M,lwd=3,lty=2,col=graycol)
points(lo.means$M,pch=19,cex=1.5,col=graycol)

arrows(seq(from=1,5,by=1),hi.means$CI.lower,
       seq(from=1,5,by=1),hi.means$CI.upper,
       lwd =  2,col="black",angle=90,code=3,length=.1)

arrows(seq(from=1,5,by=1),lo.means$CI.lower,
       seq(from=1,5,by=1),lo.means$CI.upper,
       lwd =  2,col=graycol,angle=90,code=3,length=.1)

mtext(text=c("NP3","V3","V2","V1","Post-V1"),side=1,line=.75,at=c(1,2,3,4,4.9),cex=2)
mtext(text="Reading time [ms]",side=2,line=2.5,cex=2)
mtext(text="Region",side=1,line=2.5,cex=2)

legend(2.2,1300,lty=c(1,2),col=c("black",graycol),pch=c(16,19),
       legend=c("high interference","low interference"),cex=2,lwd=3)

dev.off()

destination <- dest
system(paste("cp ",filename,destination,sep=" "))



## table for paper
xtable(verbrts.g)

xtable(verbrts.i)

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

qq.plot(residuals(fm0))

print(dotplot(ranef(fm0, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$subject)

print(dotplot(ranef(fm0, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$item)

## comparison 1 at v3:
data1 <- subset(critdata,region=="V3")

summary(fm1 <- lmer(log(value)~ g + i + gxi + (1|subject)+(1|item),
                   data=data1))

qq.plot(residuals(fm1))

print(dotplot(ranef(fm1, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$subject)

print(dotplot(ranef(fm1, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$item)


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

qq.plot(residuals(fm2))

print(dotplot(ranef(fm2, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$subject)

print(dotplot(ranef(fm2, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$item)


## comparison 3 V1:
data3 <- subset(critdata,(region=="V1"))

summary(fm3 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=data3))

summary(fm3 <- lmer(value~ g+i+gxi+(1|subject)+(1|item),
                   data=data3))

qq.plot(residuals(fm3))

print(dotplot(ranef(fm3, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$subject)

print(dotplot(ranef(fm3, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$item)

## comparison 4 post V1 region:
data4 <- subset(critdata,(region=="postV1"))

summary(fm4 <- lmer(log(value)~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4)))

summary(fm4 <- lmer(value~ g+i+gxi+(1|subject)+(1|item),
                   data=subset(data4)))

## this model is better
summary(fm4a <- lmer(log(value)~ g+i+gxi+(1+g|subject)+(1|item),
                   data=subset(data4)))


qq.plot(residuals(fm4))

print(dotplot(ranef(fm4a, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$subject)

print(dotplot(ranef(fm4, post = TRUE),
              strip=FALSE,main="Predictions for Participants Random Effects")$item)

## save data for meta-analysis
#write(t(critdata),ncolumns=10,file="e1ensprcrit.txt")

mc.fm0 <- mcmcsamp(fm0,50000)
mc.fm1 <- mcmcsamp(fm1,50000)
mc.fm2 <- mcmcsamp(fm2,50000)
mc.fm3 <- mcmcsamp(fm3,50000)
mc.fm4 <- mcmcsamp(fm4,50000)

hpd.fm0 <- lmerHPD(mc.fm0)
hpd.fm1 <- lmerHPD(mc.fm1)
hpd.fm2 <- lmerHPD(mc.fm2)
hpd.fm3 <- lmerHPD(mc.fm3)
hpd.fm4 <- lmerHPD(mc.fm4)

coef.fm0 <- fixef(fm0)
coef.fm1 <- fixef(fm1)
coef.fm2 <- fixef(fm2)[1:4]
coef.fm3 <- fixef(fm3)
coef.fm4 <- fixef(fm4)

coefs <- c(coef.fm0,
           coef.fm1,
           #coef.fm2,
           coef.fm3,coef.fm4)

#coefs <- c(#coef.fm0,
#           #coef.fm1,
#           #coef.fm2,
#           coef.fm3,coef.fm4)

hpds <- rbind(hpd.fm0$fixef[1:4,],
              hpd.fm1$fixef[1:4,],
#              hpd.fm2$fixef[1:4,], ## no need to plot this
              hpd.fm3$fixef[1:4,],
              hpd.fm4$fixef[1:4,])

#hpds <- rbind(#hpd.fm0$fixef[1:4,],
#              #hpd.fm1$fixef[1:4,],
#              hpd.fm2$fixef[1:4,], ## no need to plot this
#              hpd.fm3$fixef[1:4,],
#              hpd.fm4$fixef[1:4,])

g.hpd <- hpds[c(2,6,10,14),]
i.hpd <- hpds[c(3,7,11,15),]

g.coefs.plot <- cbind(coefs[c(2,6,10,14)],g.hpd)
i.coefs.plot <- cbind(coefs[c(3,7,11,15)],i.hpd)

xtable(g.coefs.plot,digits=5)
xtable(i.coefs.plot,digits=5)



source("plotcoefs.R")

createPS("e1ensprHPDgram.ps")

plotcoefs(g.coefs.plot,p=4,xlab=c("NP3","V3","V1","Post-V1"),
          maintext="Experiment 1: grammaticality effect (HPD intervals)")

dev.off()

createPS("e1ensprHPDint.ps")

plotcoefs(i.coefs.plot,p=4,xlab=c("NP3","V3","V1","Post-V1"),
                    maintext="Experiment 1: interference effect (HPD intervals)")
dev.off()

system(paste("cp ","e1ensprHPDgram.ps",destination,sep=" "))
system(paste("cp ","e1ensprHPDint.ps",destination,sep=" "))


### save for meta analysis:
## grammaticality factor:
critdata$gram <- as.factor(ifelse(critdata$condition%in%c("a","b"),"gram","ungram"))

## interference factor:
critdata$int <- as.factor(ifelse(critdata$condition%in%c("a","c"),"hi","lo"))

critdata$expt <- "e1"
critdata$lang <- "english"
critdata$method <- "spr"

write.table(critdata,file="e1critdata.txt")

