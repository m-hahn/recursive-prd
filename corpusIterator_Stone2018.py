from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, section=None, partition="train", removeMarkup=True, tokenize=True):
  assert language == "german"
  with open("../stimuli/Stone_etal_2018/data_spr.txt", "r") as inFile:
     text = [x.replace('"', "").split("\t") for x in inFile.read().strip().split("\n")]
  header = text[0]
  header = dict(zip(header, range(len(header))))
  text = text[1:]
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     word = line[header['word']].split("_")
     if ord(word[-1][-1]) < 65:
        word.append(word[-1][-1])
        word[-2] = word[-2][:-1]
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

def test(language, removeMarkup=True, tokenize=True):
  return load(language, removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("german", "spr")
    next(stream) 
