
library(tidyr)
library(dplyr)
library(lme4)

models = c(
905843526,
655887140,
766978233,
502504068,
697117799,
860606598
)

data = data.frame()
for(model in models) {
   data2 = read.csv(paste("output/Bartek_BB_", model, sep=""), sep="\t")
   data2$Model = model
   data = rbind(data, data2)
}
data = data %>% filter(!grepl("OOV", RegionLSTM))
# from ~/scr/recursive-prd/BarteketalJEP2011data/master.tex
 raw.spr.data <- read.table("~/scr/recursive-prd/BarteketalJEP2011data/bb-spr06-data.txt")
 colnames(raw.spr.data) <- c("subj","expt","item","condition","roi","word","correct","RT")
 raw.spr.data$LineNumber = (1:nrow(raw.spr.data))-1

 raw.spr.data = merge(raw.spr.data, data, by=c("LineNumber"))


# This item and condition seems like it's missing a noun after the fourth word, so excluded
raw.spr.data = raw.spr.data %>% filter(!(item == 13 & condition == "f"))

raw.spr.data =  raw.spr.data %>% filter(condition %in% c("a","b","c","d","e","f"))
raw.spr.data$embedding = NA
raw.spr.data$intervention = NA

raw.spr.data[raw.spr.data$condition == "a",]$embedding = "matrix"
raw.spr.data[raw.spr.data$condition == "a",]$intervention = "none"

raw.spr.data[raw.spr.data$condition == "b",]$embedding = "matrix"
raw.spr.data[raw.spr.data$condition == "b",]$intervention = "pp"

raw.spr.data[raw.spr.data$condition == "c",]$embedding = "matrix"
raw.spr.data[raw.spr.data$condition == "c",]$intervention = "rc"

raw.spr.data[raw.spr.data$condition == "d",]$embedding = "emb"
raw.spr.data[raw.spr.data$condition == "d",]$intervention = "none"

raw.spr.data[raw.spr.data$condition == "e",]$embedding = "emb"
raw.spr.data[raw.spr.data$condition == "e",]$intervention = "pp"

raw.spr.data[raw.spr.data$condition == "f",]$embedding = "emb"
raw.spr.data[raw.spr.data$condition == "f",]$intervention = "rc"

raw.spr.data = raw.spr.data %>% filter(expt == "E1")



vb = raw.spr.data %>% filter(roi == case_when(embedding == "matrix" ~ case_when(intervention == "none" ~ 2, intervention == "pp" ~ 5, intervention == "rc" ~ 7), embedding == "emb" ~ case_when(intervention == "none" ~ 5, intervention == "pp" ~ 8, intervention == "rc" ~ 10)))

vb = vb %>% mutate(pp_rc = case_when(intervention == "rc" ~ 1, intervention == "pp" ~ -1, TRUE ~ 0))
vb = vb %>% mutate(emb_c = case_when(embedding == "matrix" ~ -1, embedding == "emb" ~ 1))
vb = vb %>% mutate(someIntervention = case_when(intervention == "none" ~ -1, TRUE ~ 1))

summary(lmer(Surprisal ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + ( 1+ pp_rc + emb_c + someIntervention|Model), data=vb)) 
# No locality effect, and facilitation in embedded sentences!
summary(lmer(Surprisal ~ pp_rc + someIntervention + (1 + pp_rc + someIntervention|item) + ( 1+ pp_rc + someIntervention|Model), data=vb %>% filter(embedding == "matrix")))
# Locality effect in embedded sentence!
summary(lmer(Surprisal ~ pp_rc + someIntervention + (1 + pp_rc + someIntervention|item) + ( 1+ pp_rc + someIntervention|Model), data=vb %>% filter(embedding != "matrix")))
# Interaction as would have been suggested by Grodner&Gibson
summary(lmer(Surprisal ~ emb_c*someIntervention + (1 + someIntervention+emb_c+someIntervention*emb_c|item) + ( 1+ someIntervention+emb_c+someIntervention*emb_c|Model), data=vb))


