import random
import subprocess
scripts = []
scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_Long.py")
scripts.append("char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_NoChars_Erasure_TrainLoss_LastAndPos12_b_Long.py")


for i in range(100):
   deletion_rate = str(random.choice([0.7, 0.8, 0.9]))
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", random.choice(scripts), "--tuning=1", "--deletion_rate="+deletion_rate]
   print(command)
   subprocess.call(command)
