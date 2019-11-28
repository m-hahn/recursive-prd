import sys
import os

PATH = "/u/scr/mhahn/reinforce-logs/results/"

logs = os.listdir(PATH)

results = []
for filen in logs:
   data = open(PATH+filen, "r").read().strip().split("\n")
   if len(data) == 2:
      continue
      data.append("-1")
   params, perform, memRate,  = data
   params = params.replace("Namespace(", "")[:-1].split(", ")
   params = [x for x in params if x.split("=")[0] in ["RATE_WEIGHT", "batchSize", "entropy_weight", "learning_rate", "momentum"]]
   memRate = memRate.replace("tensor(", "").replace(", device='cuda:0', grad_fn=<MeanBackward0>)", "")
   rate = float(params[0].split("=")[1])
   results.append((rate, round(float(perform),4), round(float(memRate),4), " ".join(params), filen))
results = sorted(results, reverse=True)
lastR = None
for r in results:
   if lastR is not None and lastR[0] != r[0]:
      print("-----------")
   print("\t".join([str(x) for x in r]))
   lastR = r
