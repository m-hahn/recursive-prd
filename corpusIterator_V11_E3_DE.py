from paths import WIKIPEDIA_HOME
import random
 
import codecs
def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "german"
  with codecs.open("/u/scr/mhahn/recursive-prd/VSLK_LCP/E3_DE_SPR/data/e3desprdata.txt", "r", "iso-8859-1") as inFile:
     text = [x.strip().replace("   ", " ").replace("  ", " ").split(" ") for x in inFile.read().strip().split("\n")]
  header = ["subj","expt","item","condition","position","word","RT","similarity","grammaticality"] # according to VSLK_LCP/E3_DE_SPR/analysis/rcode/e3_de_spr.R
  header = dict(zip(header, range(len(header))))
  chunk = []

  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     print(line)
     assert len(line) == len(header)
     words = line[header['word']]
     words = words.split("_")
     for word in words:

        if tokenize:
            if word[-1] in [".", ","]:
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

  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)



if __name__ == '__main__':
    stream = test("german")
    next(stream)

