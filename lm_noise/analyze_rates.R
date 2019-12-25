data = read.csv("perWordRates.tsv", sep="\t")


library(tidyr)
library(dplyr)
library(ggplot2)






data_ = data %>% filter(filenum == 216454325, position<20)

ggplot(data_, aes(x=position, y=score, group=counter, color=counter)) + geom_line() + facet_wrap(~word)


ggplot(data_, aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~counter)

#ggplot(data_, aes(x=position, y=score, group=word, color=word)) + geom_smooth() + facet_wrap(~counter)

ggplot(data %>% filter(counter==200), aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~rate+performance+version+filenum)

