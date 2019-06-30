# PROBLEM the conditions are not matched for length

library(tidyr)
library(dplyr)
library(lme4)

# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.table("~/scr/recursive-prd/BarteketalJEP2011data/gg-spr06-data.txt")
 colnames(raw.spr.data) <- c("subj", "expt", "item", "condition", "roi", "word", "RT", "embedding", 
     "intervention")


 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

data = read.csv("output/Bartek_GG_8066636", sep="\t")
data = data %>% filter(RegionLSTM != "OOV")
 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

vb = raw.spr.data %>% filter(roi == case_when(embedding == "mat" ~ case_when(intervention == "none" ~ 2, intervention == "pp" ~ 5, intervention == "rc" ~ 7), embedding == "emb" ~ case_when(intervention == "none" ~ 5, intervention == "pp" ~ 8, intervention == "rc" ~ 10)))

vb = vb %>% mutate(pp_rc = case_when(intervention == "rc" ~ 1, intervention == "pp" ~ -1, TRUE ~ 0))
vb = vb %>% mutate(emb_c = case_when(embedding == "mat" ~ -1, embedding == "emb" ~ 1))
vb = vb %>% mutate(someIntervention = case_when(intervention == "none" ~ -1, TRUE ~ 1))

summary(lmer(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item), data=vb)) # finds RC >> PP >> none
summary(lmer(Surprisal ~ pp_rc * emb_c + (1 + pp_rc + emb_c|item), data=vb)) # finds RC more difficult, but no diff between matrix and emb


