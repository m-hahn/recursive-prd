from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

itos = ["n", "v", "p", "c", "."]

     

def load(language, partition="train", removeMarkup=True, tokenize=True, d=0.2, m=0.5, r=0.5, s=0.0):

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
        assert nt in itos
        return [nt]

  chunk = []
  for _ in range(100000 if partition == "train" else 1000):
     for x in sample("s"):
         chunk.append(x)
     if len(chunk) > 100000:
         yield chunk[:100000]
         chunk = chunk[100000:]
  yield chunk

def test(language, removeMarkup=True, tokenize=True):
  return load(language, removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("dutch")
    print(next(stream) )
