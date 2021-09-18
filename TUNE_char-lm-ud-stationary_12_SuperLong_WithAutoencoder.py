import random
import subprocess
scripts = []

#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short.py")
#scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination.py")
scripts.append("char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq.py")

for i in range(100):
   deletion_rate = str(random.choice([0.4, 0.45, 0.5, 0.55])) #[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]))
   command = ["/u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python", random.choice(scripts), "--tuning=1", "--deletion_rate="+deletion_rate]
   print(command)
   subprocess.call(command)
