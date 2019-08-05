from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

itos = ["p", "c", "."]
itos += ["v"+str(i) for i in range(5)]
itos += ["n"+str(i) for i in range(5)]

d=0.2
m=0.5
r=0.5
s=0.8 # 0.8 English, 0.0 for German. Can use 0.2 instead to match Entropy with English?
# It seems that log_beta == 2.0 or 3.0 broadly produces the effects.

def sample(nt):
     if nt == "s":
        index = str(random.randint(0,4))
        return sample("np"+index) + sample("v"+index) + ["."]
     elif nt.startswith("np"):
        c = random.random()
        index = nt[2:]
        if c < 1-m:
            return sample("n"+index)
        elif c < 1-m+m*r:
            return sample("n"+index)+sample("rc")
        else:
            return sample("n"+index)+sample("pp")
     elif nt == "pp":
         index = str(random.randint(0,4))       
         return sample("p") + sample("np"+index)
     elif nt == "rc":
         index = str(random.randint(0,4))       
         if random.random() < s:
            return sample("c") + sample("v"+index) + sample("np"+index)
         else:
            return sample("c") + sample("np"+index) + sample("v"+index)
     else:
        assert nt in itos, nt
        return [nt]

     

def load(language, partition="train", removeMarkup=True, tokenize=True):


  chunk = []
  for _ in range(100000 if partition == "train" else 10000):
     for x in sample("s"):
         chunk.append(x)
     if len(chunk) > 100000:
         yield chunk[:100000], None
         chunk = chunk[100000:]
  yield chunk, None

def training(language, removeMarkup=True, tokenize=True):
  return load(language, partition = 'train', removeMarkup=removeMarkup, tokenize=tokenize)

def dev(language, removeMarkup=True, tokenize=True):
  return load(language, partition = 'dev', removeMarkup=removeMarkup, tokenize=tokenize)

def addRegions(l):
   return ["_f"+x for x in l]

def test(language, removeMarkup=True, tokenize=True):
  chunk = []
  regions = []
  for _ in range(1000):
     index1 = str(random.randint(0,4))
     index2 = str(random.randint(0,4))
     index3 = str(random.randint(0,4))

     chunk += ["n"+index1, "c", "n"+index2, "c", "n"+index3, "v"+index3, "v"+index2, "v"+index1, "."]

     regions += ["n0", "c0", "n1", "c1", "n2", "v3", "v2", "v1g", ".g"]
     for _ in range(10): # ADD FILLERS
        l = sample("s")
        chunk += l
        regions += addRegions(l)
     index1 = str(random.randint(0,4))
     index2 = str(random.randint(0,4))
     index3 = str(random.randint(0,4))

     chunk += ["n"+index1, "c", "n"+index2, "c", "n"+index3, "v"+index3, "v"+index1, "."]
     regions += ["n0", "c0", "n1", "c1", "n2", "v3", "v1u", ".u"]
     for _ in range(10): # ADD FILLERS
        l = sample("s")
        chunk += l
        regions += addRegions(l)
  yield chunk, regions




if __name__ == '__main__':
    stream = test("dutch")
    print(next(stream) )
