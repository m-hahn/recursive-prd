# /u/nlp/anaconda/main/anaconda3/envs/py37-mhahn/bin/python GENERATE_BBB_char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py --language=english --load-from=905843526






print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model



#data = read.csv("~/scr/TMP/bartek2.txt", sep="\t")
#names(data) = c("Embedding", "Intervention", "Category", "Count")
#data$IsV = (data$Category == " vbd")
#summary(glm(IsV ~ Embedding + Intervention, data=data)) # VBD less predicted in matrix, more in PP, less in RC
#savehistory(file = ".Rhistory")


import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from", dest="load_from", type=str)

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



import corpusIterator_Bartek_BB_Py37



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt", "english" : "vocabularies/english-wiki-word-vocab-50000.txt"}[args.language]

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])

itos_complete = ["SOS", "EOS", "OOV"] + itos
stoi_complete = dict([(itos_complete[i],i) for i in range(len(itos_complete))])


with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])


itos_chars_total = ["SOS", "EOS", "OOV"] + itos_chars


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

word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)
softmax = torch.nn.Softmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')

modules = [rnn, output, word_embeddings]


character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()

char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, args.word_embedding_size).cuda()

char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total))


modules += [character_embeddings, char_composition, char_composition_output, char_decoder_rnn, char_decoder_output]
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
with open("/u/scr/mhahn/FAIR18/english-wiki-word-vocab_POS.txt", "r") as dictIn:
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
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__.replace("GENERATE_BBB_", "")+"_code_"+str(args.load_from)+".txt")
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
      numerified_chars = []
      line_numbers = []
      region_list = []
      for chunk, chunk_line_numbers, regions in data:
       for char, linenum, region in zip(chunk, chunk_line_numbers, regions):
         count += 1
         numerified.append((stoi[char]+3 if char in stoi else 2))
         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])
         line_numbers.append(linenum)
         region_list.append(region)

       assert len(region_list) == len(numerified)

       if len(numerified) > (1*args.sequence_length):
         sequenceLengthHere = args.sequence_length

         cutoff = int(len(numerified)/(1*sequenceLengthHere)) * (1*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerifiedCurrent_chars = numerified_chars[:cutoff]

         for i in range(len(numerifiedCurrent_chars)):
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:15] + [1]
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(16-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
         numerified_chars = numerified_chars[cutoff:]
         regionsHere = region_list[:cutoff]
 
         line_numbersCurrent = line_numbers[:cutoff]
         line_numbers = line_numbers[cutoff:]
         region_list = region_list[cutoff:]

         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(1, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(1, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

         line_numbersCurrent = torch.LongTensor(line_numbersCurrent).view(1, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], numerifiedCurrent_chars[i], line_numbersCurrent[i], regionsHere[i*sequenceLengthHere:(i+1)*sequenceLengthHere]
         hidden = None
       else:
         print("Skipping")


completeData = []



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())


def doGeneration(outHere, hiddenHere):
   result = ["" for _ in range(100)]
   outHere = outHere.expand(1, 100, -1)
   hiddenHere = [x.expand(-1, 100, -1).contiguous() for x in hiddenHere]
   GENERATING_LENGTH = 1
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

      numerified_chars = [([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char]) for char in nextWordStrings]
      numerified_chars = [x[:15] + [1] for x in numerified_chars]
      numerified_chars = [x+([0]*(16-len(x))) for x in numerified_chars]

      numerified_chars = torch.LongTensor(numerified_chars).view(1, 100, 1, 16).transpose(0,1).transpose(1,2).cuda()

      embedded_chars = numerified_chars.transpose(0,2).transpose(2,1)
      embedded_chars = embedded_chars.contiguous().view(16, -1)
      _, embedded_chars = char_composition(character_embeddings(embedded_chars), None)
      embedded_chars = embedded_chars[0].view(2, 1, 100, args.char_enc_hidden_dim)
      embedded_chars = char_composition_output(torch.cat([embedded_chars[0], embedded_chars[1]], dim=2))
      embedded = word_embeddings(nextWord)
      #print(embedded.size())
      #print(embedded_chars.size())
      embedded = torch.cat([embedded, embedded_chars], dim=2)
      #print(embedded.size())
      outHere, hiddenHere = rnn_drop(embedded, hiddenHere)
#   print(result)
   return result, entropy

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
          beginning_chars = zeroBeginning_chars
      elif hidden is not None:
          hidden = tuple([Variable(x.data).detach() for x in hidden])

      numeric, numeric_chars, lineNumbers, regionNames = numericAndLineNumbers


      numeric = torch.cat([beginning, numeric], dim=0)

      numeric_chars = torch.cat([beginning_chars, numeric_chars], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
      beginning_chars = numeric_chars[numeric_chars.size()[0]-1].view(1, args.batchSize, 16)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_chars = Variable(numeric_chars[:-1], requires_grad=False)
      target_tensor_chars = Variable(numeric_chars[:-1], requires_grad=False)

      embedded_chars = input_tensor_chars.transpose(0,2).transpose(2,1)
      embedded_chars = embedded_chars.contiguous().view(16, -1)
      _, embedded_chars = char_composition(character_embeddings(embedded_chars), None)
      embedded_chars = embedded_chars[0].view(2, args.sequence_length, args.batchSize, args.char_enc_hidden_dim)
      #print(embedded_chars.size())

      embedded_chars = char_composition_output(torch.cat([embedded_chars[0], embedded_chars[1]], dim=2))
      #print(embedded_chars.size())

    #  print(word_embeddings)
      #if train and (embedding_full_dropout_prob is not None):
      #   embedded = embedded_dropout(word_embeddings, input_tensor, dropout=embedding_full_dropout_prob, scale=None) #word_embeddings(input_tensor)
      #else:
      embedded = word_embeddings(input_tensor)
      #print(embedded.size())
#      print("=========")
#      print(numeric[:,5])
#      print(embedded[:,5,:].mean(dim=1)[numeric[:-1,5] == 3])
#      print(embedded_chars[:,5,:].mean(dim=1)[numeric[:-1,5] == 3])
      embedded = torch.cat([embedded, embedded_chars], dim=2)
      #print(embedded.size())

      out = [None for _ in regionNames]
      generated = [None for _ in regionNames]
      entropies = [None for _ in regionNames]
      for i in range(len(regionNames)):
          out[i], hidden = rnn_drop(embedded[i:i+1], hidden)
#          print(regionNames[i])
          condition, roi = regionNames[i].split("_")
#vb = raw.spr.data %>% filter(roi == case_when(embedding == "matrix" ~ case_when(intervention == "none" ~ 2, intervention == "pp" ~ 5, intervention == "rc" ~ 7), embedding == "emb" ~ case_when(intervention == "none" ~ 5, intervention == "pp" ~ 8, intervention == "rc" ~ 10)))
          if generateAfterNext:
              generated[i], entropies[i] = doGeneration(out[i], hidden)
              print(embedding, "\t", intervention, "\t", "\t".join([str(y) for y in (countList([posDictMax.get(x[1:], "UNK") for x in generated[i]])[0])]), file=sys.stderr)


          if condition == "a":
            embedding = "matrix"
            intervention = "none"
            reg = "1"
          elif condition == "b":
            embedding = "matrix"
            intervention = "pp"
            reg = "4"
          elif condition == "c":
            embedding = "matrix"
            intervention = "rc"
            reg = "6"
          elif condition == "d":
            embedding = "emb"
            intervention = "none"
            reg = "4"
          elif condition == "e":
            embedding = "emb"
            intervention = "pp"
            reg = "7"
          elif condition == "f":
            embedding = "emb"
            intervention = "rc"
            reg = "9"
          else:
            embedding = None
            intervention = None
            reg = "NONE"
          if roi == reg:
            print("GENERATING")
            generateAfterNext = True
          else:
            generateAfterNext = False

#      if train:
#          out = dropout(out)

      out = torch.cat(out, dim=0)

      logits = output(out) 
      log_probs = logsoftmax(logits)
   #   print(logits)
  #    print(log_probs)
 #     print(target_tensor)

      
      lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
      losses = lossTensor.data.cpu().numpy()




      for i in range(0,args.sequence_length): #range(1,maxLength+1): # don't include i==0
         j = 0
         numericCPU = numeric.cpu().data.numpy()
         lineNum = int(lineNumbers[i][j])

         print (i, itos_complete[numericCPU[i+1][j]], losses[i][j], lineNum, entropies[i]) #, countList(generated[i]), countList([posDictMax.get(x[1:], "UNK") for x in generated[i]]) if generated[i] is not None else None)
         if generated[i] is not None:
            print (countList([posDictMax.get(x[1:], "UNK") for x in generated[i]]) if generated[i] is not None else None)
            print (countList(generated[i]))


         while lineNum >= len(completeData):
             completeData.append([[], 0, None])
         completeData[lineNum][0].append(itos_complete[numericCPU[i+1][j]])
         completeData[lineNum][1] += losses[i][j]
         assert completeData[lineNum][2] == None
         completeData[lineNum][2] = countDict([posDictMax.get(x[1:], "UNK") for x in generated[i]]).get("vbd", 0) if generated[i] is not None else None
        
      return None, target_tensor.view(-1).size()[0]




import time

testLosses = []

if True:
   rnn_drop.train(False)


   test_data = corpusIterator_Bartek_BB_Py37.load(args.language, tokenize=True, forGeneration=True)
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


with open("outputGeneration/"+"Bartek_BB"+"_"+args.load_from, "w") as outFile:
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



