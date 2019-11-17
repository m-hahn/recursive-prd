import os
PATH = "/u/scr/mhahn/recursive-prd/actr/"
files = os.listdir(PATH)

print(files)
results = []
for file_ in files:
  args, losses = open(PATH+file_, "r").read().strip().split("\n")
  losses = losses.split(" ")
  results.append((-min([float(x) for x in losses]), len(losses), args))
for r in sorted(results):
   print("\t".join([str(x) for x in r]))

