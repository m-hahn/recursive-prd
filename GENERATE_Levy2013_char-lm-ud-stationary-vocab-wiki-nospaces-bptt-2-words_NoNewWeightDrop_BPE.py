# Running this:
#~/python-py37-mhahn GENERATE_Levy2013_char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop_BPE.py --load-from=717997437 --section=1a > ~/scr/TMP/tmp.txt



print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="russian")
parser.add_argument("--load-from", dest="load_from", type=str)
parser.add_argument("--section", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([1]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)

model = "REAL_REAL"

import math

args=parser.parse_args()

#if "MYID" in args.save_to:
#   args.save_to = args.save_to.replace("MYID", str(args.myID))

#assert "word" in args.save_to, args.save_to

print(args)

assert args.section == "1a"

import corpusIterator_Levy2013


def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x


words_vocab_path = f"/u/scr/mhahn/FAIR18/WIKIPEDIA/{args.language.lower()}/{args.language.lower()}-wiki-word-vocab.txt"
bpe_vocab_path = f"/u/scr/mhahn/FAIR18/WIKIPEDIA/{args.language.lower()}/{args.language.lower()}-wiki-word-vocab_BPE_50000_Parsed.txt"

vocab_size_considered = 3*int(1e6)

itos_words = [None for _ in range(vocab_size_considered)]
i2BPE = [None for _ in range(vocab_size_considered)]
with open(words_vocab_path, "r") as inFile:
  with open(bpe_vocab_path, "r") as inFileBPE:
     for i in range(vocab_size_considered):
        if i % 50000 == 0:
           print(i)
        word = next(inFile).strip().split("\t")
        bpe = next(inFileBPE).strip().split("\t")
        itos_words[i] = word[0]
        i2BPE[i] = bpe[0].split("@@ ")
stoi_words = dict([(itos_words[i],i) for i in range(len(itos_words))])

itos = []
with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
     itos = [x for x in inFile.read().strip().split("\n")]
itos += [x+"</w>" for x in itos]

with open(f"/u/scr/mhahn/FAIR18/WIKIPEDIA/{args.language.lower()}/{args.language.lower()}-wiki-word-vocab_BPE_50000.txt", "r") as inFile:
     itos += [x.replace(" ",  "") for x in inFile.read().strip().split("\n") if not x.startswith("#version: ")]
assert len(itos) > 50000, len(itos)

#print(itos)
#print(stoi)
#quit()
assert len(itos_words) > 50000
stoi = dict([(itos[i],i) for i in range(len(itos))])

itos_complete = ["SOS", "EOS", "OOV"] + itos
stoi_complete = dict([(itos_complete[i],i) for i in range(len(itos_complete))])


#with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
#     itos_chars = [x for x in inFile.read().strip().split("\n")]
#stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])


#itos_chars_total = ["<SOS>", "<EOS>", "OOV"] + itos_chars


import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop


rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = rnn #WeightDrop(rnn, layer_names=[(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)
softmax = torch.nn.Softmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')

modules = [rnn, output, word_embeddings]


#character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()

#char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
#char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, args.word_embedding_size).cuda()
#
#char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
#char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total))
#
#
#modules += [character_embeddings, char_composition, char_composition_output, char_decoder_rnn, char_decoder_output]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

#named_modules = {"rnn" : rnn, "output" : output, "word_embeddings" : word_embeddings, "optim" : optim}


#   state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
#   torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")


posDict = {}
with open("/u/scr/mhahn/FAIR18/WIKIPEDIA/russian/russian-wiki-word-vocab_POS.txt", "r") as dictIn:
    for line in dictIn:
        line = line.strip().split("\t")
        line[2] = int(line[2])
        if line[2] < 100:
           print(len(posDict))
           break
        if line[0] not in posDict:
           posDict[line[0]] = {}
        if line[1] not in posDict[line[0]]:
            posDict[line[0]][line[1]] = 0
        posDict[line[0]][line[1]] += 1

posDictMax = {}
for word, entry in posDict.items():
    maxCount = max([entry[x] for x in entry])
    bestPOS = [x for x in entry if entry[x] == maxCount][0]
    posDictMax[word] = bestPOS
        
if args.load_from is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__.replace("GENERATE_Levy2013_", "")+"_code_"+str(args.load_from)+".txt")
  for i in range(len(checkpoint["components"])):
      modules[i].load_state_dict(checkpoint["components"][i])
else:
  assert False

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout

positionHere = 0

def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
#      numerified_chars = []
      line_numbers = []
      region_list = []
      for chunk, chunk_line_numbers, regions in data:
       for char, linenum, region in zip(chunk, chunk_line_numbers, regions):
         count += 1
         char_i = stoi_words.get(char, -1)
         if char_i == -1:
            numerified.append(2)
            line_numbers.append(linenum)
            region_list.append(region)
         else:
            bpes = i2BPE[stoi_words[char]]
            if not bpes[-1].endswith("</w>"):
                bpes[-1] += "</w>"
            for bpe in bpes:
               numerified.append(stoi.get(bpe, -1)+3) 
               line_numbers.append(linenum)
               region_list.append(region)
       assert len(region_list) == len(numerified)

       if len(numerified) > (1*args.sequence_length):
         sequenceLengthHere = args.sequence_length

         cutoff = int(len(numerified)/(1*sequenceLengthHere)) * (1*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
#         numerifiedCurrent_chars = numerified_chars[:cutoff]

#         for i in range(len(numerifiedCurrent_chars)):
#            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:15] + [1]
#            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(16-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
#         numerified_chars = numerified_chars[cutoff:]
         regionsHere = region_list[:cutoff]
 
         line_numbersCurrent = line_numbers[:cutoff]
         line_numbers = line_numbers[cutoff:]
         region_list = region_list[cutoff:]

         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(1, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
#         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(args.batchSize, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

         line_numbersCurrent = torch.LongTensor(line_numbersCurrent).view(1, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], line_numbersCurrent[i], regionsHere[i*sequenceLengthHere:(i+1)*sequenceLengthHere]
         hidden = None
       else:
         print("Skipping")


completeData = []



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

#zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())


def doGeneration(outHere, hiddenHere, preceding):
   result = ["" for _ in range(100)]
   outHere = outHere.expand(1, 100, -1)
   hiddenHere = [x.expand(-1, 100, -1).contiguous() for x in hiddenHere]
   GENERATING_LENGTH = 10
   for l in range(GENERATING_LENGTH):
      logits = output(outHere)
      probs = softmax(logits)
      
#      print(probs.size())
      if l == 0:
           entropy = -float((probs[0,0] * torch.log(probs[0,0])).sum())

      dist = torch.distributions.Categorical(probs=probs)
       
      nextWord = (dist.sample())
#      print([x for x in nextWord.cpu().numpy()[0]])

      nextWordStrings = [itos_complete[x] for x in nextWord.cpu().numpy()[0]]
      for i in range(100):
         result[i] += " "+nextWordStrings[i]
      if l == GENERATING_LENGTH-1:
         break


      embedded = word_embeddings(nextWord)
      outHere, hiddenHere = rnn_drop(embedded, hiddenHere)
   result2 = []
   preceding = preceding.replace(" ", "").replace("</w>", " ")
   for r in result:
      r = r.replace(" ", "").replace("</w>", " ")
     
      print(preceding+r)
      result2.append(r.split(" "))
   return result2, entropy

def countList(x):
   if x is None:
     return None
   result = {}
   for y in x:
     result[y] = result.get(y,0)+1
   return sorted(list(result.items()), key=lambda x:-x[1])

def countDict(x):
   if x is None:
     return None
   result = {}
   for y in x:
     result[y] = result.get(y,0)+1
   return result


resultsByConditionAndRegion = {}
generateAfterNext = False
def forward(numericAndLineNumbers, train=True, printHere=False):
      global generateAfterNext
      global hidden
      global beginning
      global beginning_chars
      global embedding
      global intervention
      if hidden is None:
          hidden = None
          beginning = zeroBeginning
#          beginning_chars = zeroBeginning_chars
      elif hidden is not None:
          hidden = tuple([Variable(x.data).detach() for x in hidden])

      numeric, lineNumbers, regionNames = numericAndLineNumbers
      numeric = torch.cat([beginning, numeric], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)

      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      embedded = word_embeddings(input_tensor)

      out = [None for _ in regionNames]
      generated = [None for _ in regionNames]
      entropies = [None for _ in regionNames]
      for i in range(len(regionNames)):
          out[i], hidden = rnn_drop(embedded[i:i+1], hidden)
          print(regionNames[i])

          condition, region = regionNames[i].split("_")

          if itos_complete[numeric[i][0]].endswith(">"):



              if generateAfterNext:
                  preceding = " ".join([itos_complete[numeric[j][0]] for j in range(i-4, i+1)])
                  generated[i], entropies[i] = doGeneration(out[i], hidden, preceding)
                  print(generated[i][0])
                  print((condition, region))
                  poss = countList([posDictMax.get(x[0], "UNK") for x in generated[i]])
                  print(poss, file=sys.stderr)
                  words = countList([x[0] for x in generated[i]])
                  print(words, file=sys.stderr)
                  if (condition, region) not in resultsByConditionAndRegion:
                      resultsByConditionAndRegion[(condition, region)] = []
                  

                  resultsByConditionAndRegion[(condition, region)].append((words, poss, generated[i], preceding.replace(" ", "").replace("</w>", " ")))
    
              if ((region == "V0" and condition in ["A", "D"]) or (region == "N1" and condition in ["B", "C"])):
                print("GENERATING", int(lineNumbers[i][0]), itos_complete[numeric[i][0]])
                generateAfterNext = True
              else:
                generateAfterNext = False
   


      out = torch.cat(out, dim=0)

      logits = output(out) 
      log_probs = logsoftmax(logits)

      
      lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
      losses = lossTensor.data.cpu().numpy()




      for i in range(0,args.sequence_length): #range(1,maxLength+1): # don't include i==0
         j = 0
         numericCPU = numeric.cpu().data.numpy()
         lineNum = int(lineNumbers[i][j])

         print (i, itos_complete[numericCPU[i+1][j]], losses[i][j], lineNum, entropies[i]) #, countList(generated[i]), countList([posDictMax.get(x[1:], "UNK") for x in generated[i]]) if generated[i] is not None else None)
         if generated[i] is not None:
            print (countList([posDictMax.get(x[0], "UNK") for x in generated[i]]) if generated[i] is not None else None)
            print (countList([x[0] for x in generated[i]]))


         while lineNum >= len(completeData):
             completeData.append([[], 0, None])
         completeData[lineNum][0].append(itos_complete[numericCPU[i+1][j]])
         completeData[lineNum][1] += losses[i][j]
         assert completeData[lineNum][2] == None or generated[i] == None, lineNum
 #        if generated[i] is not None:
#            completeData[lineNum][2] = countDict([posDictMax.get(x[0], "UNK") for x in generated[i]]).get("vbd", 0) if generated[i] is not None else None
        
      return None, target_tensor.view(-1).size()[0]




import time

testLosses = []

if True:
   rnn_drop.train(False)


   test_data = corpusIterator_Levy2013.load(args.language, section=args.section, tokenize=True, forGeneration=True)
   print("Got data")
   test_chars = prepareDatasetChunks(test_data, train=False)


     
   test_loss = 0
   test_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
          numeric = next(test_chars)
       except StopIteration:
          print("End of corpus")
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       assert numberOfCharacters > 0
       test_char_count += numberOfCharacters
   testLosses.append(test_loss/test_char_count)
   print(testLosses)


with open("outputGeneration/"+"Levy2013"+"_"+args.load_from, "w") as outFile:
   print("\t".join(["LineNumber", "RegionLSTM", "Surprisal", "VBD_Generated"]), file=outFile)
   for num, entry in enumerate(completeData):
     words = "".join(entry[0])
     if len(words) == 0:
        words= "NONE"
     print("\t".join([str(x) for x in [num, words, entry[1], entry[2]]]), file=outFile)

#
#data = read.csv("~/scr/TMP/bartek2.txt", sep="\t")
#names(data) = c("Embedding", "Intervention", "Category", "Count")
#data$IsV = (data$Category == " vbd")
#summary(glm(IsV ~ Embedding + Intervention, data=data))
#savehistory(file = ".Rhistory")
#


for condition, region in resultsByConditionAndRegion:
    print("====================================")
    for entry in resultsByConditionAndRegion[(condition, region)]:
       wordss, poss, continuations, preceding = entry
       print(condition, region, wordss)
       print(condition, region, poss)
       print("\n".join([condition+" "+region+" "+preceding+" "+(" ".join(x)) for x in continuations]))

