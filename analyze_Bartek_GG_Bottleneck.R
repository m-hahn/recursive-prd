library(tidyr)
library(dplyr)
library(lme4)

# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.table("../../recursive-prd/BarteketalJEP2011data/gg-spr06-data.txt")
 colnames(raw.spr.data) <- c("subj", "expt", "item", "condition", "roi", "word", "RT", "embedding", 
     "intervention")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

modelsTable = read.csv("results/models_bottlenecked_english", sep="\t")

models = modelsTable$ID
datModel = data.frame()
for(model in models) {
   datModel2 = tryCatch(read.csv(paste("output/Bartek_GG_", model, sep=""), sep="\t") %>% mutate(Model = model), error=function(q) 1)
   if(datModel2 != 1) {
     cat(model,"\n")
     datModel = rbind(datModel, datModel2)
   }
}

modelsTable = modelsTable %>% mutate(Model=ID, ModelPerformance = Surprisal) %>% mutate(ID=NULL, Surprisal=NULL)
datModel = merge(datModel, modelsTable, by=c("Model"))
datModel = datModel %>% filter(RegionLSTM != "OOV")

 raw.spr.data = merge(raw.spr.data, datModel, by=c("LineNumber"))


vb = raw.spr.data %>% filter(roi == case_when(embedding == "mat" ~ case_when(intervention == "none" ~ 2, intervention == "pp" ~ 5, intervention == "rc" ~ 7), embedding == "emb" ~ case_when(intervention == "none" ~ 5, intervention == "pp" ~ 8, intervention == "rc" ~ 10)))

vb = vb %>% mutate(pp_rc = case_when(intervention == "rc" ~ 1, intervention == "pp" ~ -1, TRUE ~ 0))
vb = vb %>% mutate(emb_c = case_when(embedding == "mat" ~ -1, embedding == "emb" ~ 1))
vb = vb %>% mutate(someIntervention = case_when(intervention == "none" ~ -1, TRUE ~ 1))

vb = vb %>% mutate(ModelPerformance.C=ModelPerformance-mean(ModelPerformance), LogBeta.C=LogBeta-mean(LogBeta))
# TODO why are there duplicates???
vb_ = unique(vb %>% select(Surprisal, pp_rc, emb_c, someIntervention, item, Model, intervention, embedding, ModelPerformance.C, Script, LogBeta.C, LogBeta, Memory, ModelPerformance))


library(ggplot2)

plot = ggplot(data=vb_ %>% group_by(intervention, Model, embedding) %>% summarise(Surprisal=mean(Surprisal)), aes(x=intervention, y=Surprisal, group=Model, color=Model)) + geom_line() + facet_wrap(~embedding)
ggsave(plot, file="figures/bartek_gg_bottleneck.pdf")

plot = ggplot(data=vb_ %>% filter(embedding == "mat") %>% group_by(ModelPerformance, intervention, Model, embedding) %>% summarise(Surprisal=mean(Surprisal)), aes(x=intervention, y=Surprisal, group=Model, color=Model)) + geom_line() + facet_wrap(~ModelPerformance)
ggsave(plot, file="figures/bartek_gg_bottleneck_matrix.pdf")

plot = ggplot(data=vb_ %>% filter(embedding == "emb") %>% group_by(ModelPerformance, intervention, Model, embedding) %>% summarise(Surprisal=mean(Surprisal)), aes(x=intervention, y=Surprisal, group=Model, color=Model)) + geom_line() + facet_wrap(~ModelPerformance)
ggsave(plot, file="figures/bartek_gg_bottleneck_emb.pdf")





summary(lmer(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + (1 + pp_rc + emb_c + someIntervention|Model), data=vb_)) # finds RC >> PP >> none
#summary(lmer(Surprisal ~ pp_rc * emb_c + (1 + pp_rc + emb_c|item), data=vb)) # finds RC more difficult, but no diff between matrix and emb


summary(lmer(Surprisal ~ pp_rc + emb_c + LogBeta.C*someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + (1|Model), data=vb_))

library(brms)

model = brm(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + (1 + pp_rc + emb_c + someIntervention|Model), data=vb_)

summary(model)





