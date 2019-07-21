from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "english"
  with open("/u/scr/mhahn/recursive-prd/BarteketalJEP2011data/bb-spr06-data.txt", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  #print(text)
  header = ["subj","expt","item","condition","roi","word","correct","RT"] # bb-spr-dataprep.Rnw
  header = dict(zip(header, range(len(header))))
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     word = line[header['word']]
     if line[header["correct"]] != "-":
        print("SPECIAL", line)
        continue
#     words = words[1:-1].split("_")
     if True:
#        if word in ["N", "Y"]:
 #          continue

        if tokenize:
            if word[-1] in [".", ",", "?"]:
               chunk.append(word[:-1].lower())
               chunk.append(word[-1].lower())
               chunk_line_numbers.append(linenum)
               chunk_line_numbers.append(linenum)
            else:
               chunk.append(word.lower())
               chunk_line_numbers.append(linenum)
        else:
           for char in " "+word:
               chunk.append(char.lower())
               chunk_line_numbers.append(linenum)

#  print(chunk)
  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True, tokenize=True):
  return load(language, "test", removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("english")
    next(stream) 
