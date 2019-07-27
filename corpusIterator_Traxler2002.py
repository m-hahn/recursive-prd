from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, section, partition="train", removeMarkup=True, tokenize=True):
  assert section in ["expt1", "expt2", "expt3"]
  assert language == "english"
  with open("stimuli/traxler_etal_2002/expt1-tokenized.tsv", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header = text[0]
  header = dict(zip(header, range(len(header))))
  text = text[1:]
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     word = line[header['Word']].split("_")
     if True:

        if tokenize:
           for word_ in word:
               chunk.append(word_.lower())
               chunk_line_numbers.append(linenum)
        else:
           for char in " "+(" ".join(word)):
               chunk.append(char.lower())
               chunk_line_numbers.append(linenum)
  print(chunk)
#  print(chunk)
  yield chunk, chunk_line_numbers

def test(language, section, removeMarkup=True, tokenize=True):
  return load(language, section, removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("english", "expt1")
    next(stream) 
