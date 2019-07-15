
library(tidyr)
library(dplyr)
library(lme4)

# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.table("~/scr/recursive-prd/BarteketalJEP2011data/gg-spr06-data.txt")
 colnames(raw.spr.data) <- c("subj", "expt", "item", "condition", "roi", "word", "RT", "embedding", 
     "intervention")


 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

data = read.csv("output/Bartek_GG_697117799", sep="\t")
data = data %>% filter(RegionLSTM != "OOV")
 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

vb = raw.spr.data %>% filter(roi == case_when(embedding == "mat" ~ case_when(intervention == "none" ~ 2, intervention == "pp" ~ 5, intervention == "rc" ~ 7), embedding == "emb" ~ case_when(intervention == "none" ~ 5, intervention == "pp" ~ 8, intervention == "rc" ~ 10)))

vb = vb %>% mutate(pp_rc = case_when(intervention == "rc" ~ 1, intervention == "pp" ~ -1, TRUE ~ 0))
vb = vb %>% mutate(emb_c = case_when(embedding == "mat" ~ -1, embedding == "emb" ~ 1))
vb = vb %>% mutate(someIntervention = case_when(intervention == "none" ~ -1, TRUE ~ 1))

summary(lmer(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item), data=vb)) # finds RC >> PP >> none
summary(lmer(Surprisal ~ pp_rc * emb_c + (1 + pp_rc + emb_c|item), data=vb)) # finds RC more difficult, but no diff between matrix and emb


models = c(
905843526,
655887140,
766978233,
502504068,
697117799,
860606598)

data = data.frame()

for(model in models) {
   data2 = read.csv(paste("output/Bartek_GG_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}


data = data %>% filter(RegionLSTM != "OOV")

 raw.spr.data <- read.table("~/scr/recursive-prd/BarteketalJEP2011data/gg-spr06-data.txt")
 colnames(raw.spr.data) <- c("subj", "expt", "item", "condition", "roi", "word", "RT", "embedding", 
     "intervention")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1


 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))

vb = raw.spr.data %>% filter(roi == case_when(embedding == "mat" ~ case_when(intervention == "none" ~ 2, intervention == "pp" ~ 5, intervention == "rc" ~ 7), embedding == "emb" ~ case_when(intervention == "none" ~ 5, intervention == "pp" ~ 8, intervention == "rc" ~ 10)))

vb = vb %>% mutate(pp_rc = case_when(intervention == "rc" ~ 1, intervention == "pp" ~ -1, TRUE ~ 0))
vb = vb %>% mutate(emb_c = case_when(embedding == "mat" ~ -1, embedding == "emb" ~ 1))
vb = vb %>% mutate(someIntervention = case_when(intervention == "none" ~ -1, TRUE ~ 1))

vb_ = unique(vb %>% select(Surprisal, pp_rc, emb_c, someIntervention, item, Model))

summary(lmer(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + (1 + pp_rc + emb_c + someIntervention|Model), data=vb_)) # finds RC >> PP >> none
#summary(lmer(Surprisal ~ pp_rc * emb_c + (1 + pp_rc + emb_c|item), data=vb)) # finds RC more difficult, but no diff between matrix and emb


library(brms)

model = brm(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + (1 + pp_rc + emb_c + someIntervention|Model), data=vb_)

summary(model)


#                 Estimate Est.Error l-95% CI u-95% CI Eff.Sample Rhat
#Intercept            9.33      0.40     8.54    10.13        569 1.01
#pp_rc                0.01      0.09    -0.17     0.20       1294 1.00
#emb_c               -0.04      0.07    -0.17     0.10       1826 1.00
#someIntervention     0.75      0.17     0.41     1.10       1248 1.00
#pp_rc:emb_c          0.06      0.01     0.03     0.09       4000 1.00



