from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

itos = ["n", "v", "p", "c", "."]


d=0.2
m=0.5
r=0.5
s=0.0 # 0.8 English, 0.0 for German. Can use 0.2 instead to match Entropy with English?
# It seems that log_beta == 2.0 or 3.0 broadly produces the effects.

def sample(nt):
     if nt == "s":
        return sample("np") + sample("v") + ["."]
     elif nt == "np":
        c = random.random()
        if c < 1-m:
            return sample("n")
        elif c < 1-m+m*r:
            return sample("n")+sample("rc")
        else:
            return sample("n")+sample("pp")
     elif nt == "pp":
         return sample("p") + sample("np")
     elif nt == "rc":
         if random.random() < s:
            return sample("c") + sample("v") + sample("np")
         else:
            return sample("c") + sample("np") + sample("v")
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
     chunk += list("ncncnvvv.")
     regions += ["n0", "c0", "n1", "c1", "n2", "v3", "v2", "v1g", ".g"]
     for _ in range(10): # ADD FILLERS
        l = sample("s")
        chunk += l
        regions += addRegions(l)
     chunk += list("ncncnvv.")
     regions += ["n0", "c0", "n1", "c1", "n2", "v3", "v1u", ".u"]
     for _ in range(10): # ADD FILLERS
        l = sample("s")
        chunk += l
        regions += addRegions(l)
  yield chunk, regions




if __name__ == '__main__':
    stream = test("dutch")
    print(next(stream) )
