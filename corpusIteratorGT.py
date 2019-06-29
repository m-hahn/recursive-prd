from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "english"
  with open("/u/scr/mhahn/recursive-prd/extention-items.txt", "r") as inFile:
     text = inFile.read().strip().split("\n")
  stimuli = []

  lastLabel = None
  for line in text:
     if line.startswith("#"):
         lastLabel = line[2:].split(" ")
         print(lastLabel)
         assert len(lastLabel) == 3
         label = lastLabel
         for x in label:
            if x not in stoi_labels:
              stoi_labels[x] = len(itos_labels)
              itos_labels.append(x)
         lastLabel = [stoi_labels[x] for x in label]
     elif len(line) > 1 and line[0] != "?":
         line = line.lower() + " "
         if tokenize:
           line = line.replace(".", " .")
           line = line.replace(",", " ,")
           line = line.strip().split(" ")

         stimuli.append((lastLabel, line))
  chunk = []
  chunk_line_numbers = []

  for iteration in range(10):
     random.shuffle(stimuli)
     for label, line in stimuli:
       for word in line:
          chunk.append(word)
          chunk_line_numbers.append(label[::] + [iteration])
     print(chunk)
   #  print(chunk)
  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True, tokenize=True):
  return load(language, "test", removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("german")
    char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt", "english" : "vocabularies/english-wiki-word-vocab-50000.txt"}["german"]
    
    with open(char_vocab_path, "r") as inFile:
         itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
    stoi = dict([(itos[i],i) for i in range(len(itos))])
    notIn = 0
    total = 0
    for word in next(stream):
      total += 1
      if word not in stoi:
        print(word)
        notIn += 1.0
    print(notIn / total)





