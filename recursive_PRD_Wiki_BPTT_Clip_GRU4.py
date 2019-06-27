
import torchkit.optim
import torchkit.nn, torchkit.flows, torchkit.utils
import numpy as np

import random
import sys

objectiveName = "LM"

import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="english")
parser.add_argument("--dropout_rate", type=float, default=random.choice([0.0, 0.1]))
parser.add_argument("--emb_dim", type=int, default=100)
parser.add_argument("--rnn_dim", type=int, default=512)
parser.add_argument("--rnn_layers", type=int, default=1)
parser.add_argument("--lr", type=float, default=random.choice([0.00001, 0.00002, 0.00005, 0.0001,0.0002, 0.001]))
parser.add_argument("--input_dropoutRate", type=float, default=0.0)
parser.add_argument("--batchSize", type=int, default=256)
parser.add_argument("--horizon", type=int, default=20)
parser.add_argument("--beta", type=float, default=math.exp(-random.choice([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])))
parser.add_argument("--flow_length", type=int, default=0) #random.choice([0,1]))
parser.add_argument("--flowtype", type=str, default=random.choice(["ddsf", "dsf"]))
parser.add_argument("--flow_hid_dim", type=int, default=512)
parser.add_argument("--flow_num_layers", type=int, default=2)
parser.add_argument("--myID", type=int, default=random.randint(0,10000000))
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--norm_clip", type=float, default=2.0)



args=parser.parse_args()
print(str(args))

model = "REAL"



assert args.dropout_rate <= 0.5
assert args.input_dropoutRate <= 0.5

devSurprisalTable = [None] * args.horizon


print("TESTING, NO LOGGING")


posUni = set() #[ "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"] 

posFine = set() #[ "``", ",", ":", ".", "''", "$", "ADD", "AFX", "CC",  "CD", "DT", "EX", "FW", "GW", "HYPH", "IN", "JJ", "JJR",  "JJS", "-LRB-", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS",  "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "-RRB-", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  "WDT", "WP", "WP$", "WRB", "XX" ]



deps = ["acl", "acl:relcl", "advcl", "advmod", "amod", "appos", "aux", "auxpass", "case", "cc", "ccomp", "compound", "compound:prt", "conj", "conj:preconj", "cop", "csubj", "csubjpass", "dep", "det", "det:predet", "discourse", "dobj", "expl", "foreign", "goeswith", "iobj", "list", "mark", "mwe", "neg", "nmod", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubjpass", "nummod", "parataxis", "punct", "remnant", "reparandum", "root", "vocative", "xcomp"] 

#deps = ["acl", " advcl", " advmod", " amod", " appos", " aux", " case cc", " ccompclf", " compound", " conj", " cop", " csubjdep", " det", " discourse", " dislocated", " expl", " fixed", " flat", " goeswith", " iobj", " list", " mark", " nmod", " nsubj", " nummod", " obj", " obl", " orphan", " parataxis", " punct", " reparandum", " root", " vocative", " xcomp"]

import math
from math import log, exp
from random import random, shuffle

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

import corpusIteratorWikiWords

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}



#import torch.distributions
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random

softmax_layer = torch.nn.Softmax()
logsoftmax = torch.nn.LogSoftmax()

from paths import CHAR_VOCAB_HOME


char_vocab_path = {"german" : "vocabularies/german-wiki-word-vocab-50000.txt", "italian" : "vocabularies/italian-wiki-word-vocab-50000.txt", "english" : "vocabularies/english-wiki-word-vocab-50000.txt"}[args.language]

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


import os


vocab_size = 50000


word_pos_morph_embeddings = torch.nn.Embedding(num_embeddings = len(itos)+3, embedding_dim=args.emb_dim).cuda()
outVocabSize = 3+len(itos) #+vocab_size+len(morphKeyValuePairs)+3


itos_total = ["EOS", "OOV", "SOS"] + itos #+ itos_lemmas[:vocab_size] + itos_morph
assert len(itos_total) == outVocabSize
# could also provide per-word subcategorization frames from the treebank as input???


#baseline = nn.Linear(args.emb_dim, 1).cuda()

dropout = nn.Dropout(args.dropout_rate).cuda()

rnn_both = nn.GRU(args.emb_dim, args.rnn_dim, args.rnn_layers).cuda()
for name, param in rnn_both.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

decoder = nn.Linear(args.rnn_dim,outVocabSize).cuda()
#pos_ptb_decoder = nn.Linear(128,len(posFine)+3).cuda()


components = [rnn_both, decoder, word_pos_morph_embeddings]


#           klLoss = [None for _ in inputEmbeddings]
#           logStandardDeviationHidden = hiddenToLogSDHidden(hidden[1][0])
#           sampled = torch.normal(hiddenMean, torch.exp(logStandardDeviationHidden))
#           klLoss = 0.5 * (-1 - 2 * (logStandardDeviationHidden) + torch.pow(meanHidden, 2) + torch.exp(2*logStandardDeviationHidden))
#           hiddenNew = sampleToHidden(sampled)
#           cellNew = sampleToCell(sampled)
 


hiddenToLogSDHidden = nn.Linear(args.rnn_dim, args.rnn_dim).cuda()


hiddenToLogSDHidden.weight.data.fill_(0)
hiddenToLogSDHidden.bias.data.fill_(0)



#
#weight_made = [torch.cuda.FloatTensor(args.rnn_dim, args.rnn_dim).fill_(0) for _ in range(args.flow_length)]
#for p in weight_made:
#  p.requires_grad=True
#  nn.init.xavier_normal(p)
#
#bias_made = [torch.cuda.FloatTensor(args.rnn_dim).fill_(0) for _ in range(args.flow_length)]
#for p in bias_made:
#   p.requires_grad=True
#
#weight_made_mu = [torch.cuda.FloatTensor(args.rnn_dim, args.rnn_dim).fill_(0) for _ in range(args.flow_length)]
#for p in weight_made_mu:
#   p.requires_grad=True
#
#bias_made_mu = [torch.cuda.FloatTensor(args.rnn_dim).fill_(0) for _ in range(args.flow_length)]
#for p in bias_made_mu:
#   p.requires_grad=True
#
#weight_made_sigma = [torch.cuda.FloatTensor(args.rnn_dim, args.rnn_dim).fill_(0) for _ in range(args.flow_length)]
#for p in weight_made_sigma:
#   p.requires_grad=True
#
#bias_made_sigma = [torch.cuda.FloatTensor(args.rnn_dim).fill_(0) for _ in range(args.flow_length)]
#for p in bias_made_sigma:
#   p.requires_grad=True
#
#
#
#parameters_made = [weight_made, bias_made, weight_made_mu, bias_made_mu, weight_made_sigma, bias_made_sigma]




import torchkit.nn as nn_



class BaseFlow(torch.nn.Module):
    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()





class IAF(BaseFlow):
    
    def __init__(self, dim, hid_dim, context_dim, num_layers,
                 activation=nn.ELU(), realify=nn_.sigmoid, fixed_order=False):
        super(IAF, self).__init__()
        self.realify = realify
        
        self.dim = dim
        self.context_dim = context_dim
        
        if type(dim) is int:
            self.mdl = torchkit.iaf_modules.cMADE(
                    dim, hid_dim, context_dim, num_layers, 2, 
                    activation, fixed_order)
            self.reset_parameters()
        else:
           assert False        
        
    def reset_parameters(self):
        self.mdl.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.mdl.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)
        if self.realify == nn_.softplus:
            inv = np.log(np.exp(1-nn_.delta)-1) 
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(inv,inv)
        elif self.realify == nn_.sigmoid:
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(2.0,2.0)
        
        
    def forward(self, inputs):
        x, logdet, context = inputs
        if torch.isnan(x).any():
           assert False, x
        if torch.isnan(context).any():
           assert False, context

        out, _ = self.mdl((x, context))
        if torch.isnan(out).any():
           assert False, out
        if isinstance(self.mdl, torchkit.iaf_modules.cMADE):
            mean = out[:,:,0]
            lstd = out[:,:,1]
        else:
            assert False
    
        std = self.realify(lstd)
        
        if self.realify == nn_.softplus:
            x_ = mean + std * x
        elif self.realify == nn_.sigmoid:
            x_ = (-std+1.0) * mean + std * x
        elif self.realify == nn_.sigmoid2:
            x_ = (-std+2.0) * mean + std * x
        logdet_ = nn_.sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context

 



num_ds_dim = 16 #64
num_ds_layers = 1

if args.flowtype == 'affine':
    flow = IAF
elif args.flowtype == 'dsf':
    flow = lambda **kwargs:torchkit.flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                         num_ds_layers=num_ds_layers,
                                         **kwargs)
elif args.flowtype == 'ddsf':
    flow = lambda **kwargs:torchkit.flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                          num_ds_layers=num_ds_layers,
                                          **kwargs)


#           hiddenMade = torch.nn.ReLU(torch.nn.functional.linear(sampled, weight_made * mask, bias_made))
#
#           muMade = torch.ReLU(torch.nn.functional.linear(hiddenMade, weight_made_mu * mask, bias_made_mu))
#           logSigmaMade = (torch.nn.functional.linear(hiddenMade, weight_made_sigma * mask, bias_made_sigma))
#           sigmaMade = torch.exp(logSigmaMade)





components = components + [hiddenToLogSDHidden]

context_dim = 1
flows = [flow(dim=args.rnn_dim, hid_dim=args.flow_hid_dim, context_dim=context_dim, num_layers=args.flow_num_layers, activation=torch.nn.ELU()).cuda() for _ in range(args.flow_length)]


components = components + flows




def parameters():
 for c in components:
   for param in c.parameters():
      yield param
# for q in parameters_made:
#   for p in q:
#    yield p


# yield dhWeights
# yield distanceWeights

#for pa in parameters():
#  print pa

initrange = 0.1
#word_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_u_embeddings.weight.data.uniform_(-initrange, initrange)
#pos_p_embeddings.weight.data.uniform_(-initrange, initrange)
#morph_embeddings.weight.data.uniform_(-initrange, initrange)
word_pos_morph_embeddings.weight.data.uniform_(-initrange, initrange)

decoder.bias.data.fill_(0)
decoder.weight.data.uniform_(-initrange, initrange)
#pos_ptb_decoder.bias.data.fill_(0)
#pos_ptb_decoder.weight.data.uniform_(-initrange, initrange)
#baseline.bias.data.fill_(0)
#baseline.weight.data.uniform_(-initrange, initrange)




crossEntropy = 10.0

#def encodeWord(w):
#   return stoi[w]+3 if stoi[w] < vocab_size else 1

#loss = torch.nn.CrossEntropyLoss(reduce=False, ignore_index = 0)

optimizer = torch.optim.Adam(parameters(), lr=args.lr, betas=(0.9, 0.999) , weight_decay=args.weight_decay)


import torch.cuda
import torch.nn.functional

inputDropout = torch.nn.Dropout2d(p=args.input_dropoutRate)


counter = 0


lastDevLoss = None
failedDevRuns = 0
devLosses = [] 
devMemories = []


lossModule = nn.NLLLoss()
lossModuleTest = nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)


mask1 = torch.FloatTensor([[1 if k > d else 0 for d in range(args.rnn_dim)] for k in range(args.rnn_dim)]).cuda()
mask2 = torch.FloatTensor([[1 if k < d else 0 for d in range(args.rnn_dim)] for k in range(args.rnn_dim)]).cuda()




standardNormal = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(args.rnn_dim)] for _ in range(args.horizon*args.batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(args.rnn_dim)] for _ in range(args.horizon*args.batchSize)]).cuda())
standardNormalPerStep = torch.distributions.Normal(loc=torch.FloatTensor([[0.0 for _ in range(args.rnn_dim)] for _ in range(args.batchSize)]).cuda(), scale=torch.FloatTensor([[1.0 for _ in range(args.rnn_dim)] for _ in range(args.batchSize)]).cuda())






def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      for chunk in data:
       for char in chunk:
         count += 1
         numerified.append((stoi[char]+3 if char in stoi else 1))

       if len(numerified) > (args.batchSize*args.horizon):
         sequenceLengthHere = args.horizon

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerified = numerified[cutoff:]
        
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i]
         hidden = None
       else:
         print("Skipping")



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = zeroBeginning


def doForwardPass(numeric, surprisalTable=None, doDropout=True, batchSizeHere=1):
       global counter
       global crossEntropy
       global printHere
       global devLosses
 
       global hidden
       global beginning

       if hidden is not None:
           hidden = Variable(hidden.data).detach()
       else:
           beginning = zeroBeginning

       numeric = torch.cat([beginning, numeric], dim=0)
 
       beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
 
 

       loss = 0
       wordNum = 0
       lossWords = 0
       policyGradientLoss = 0
       baselineLoss = 0

       optimizer.zero_grad()

       for c in components:
          c.zero_grad()
#       for q in parameters_made:
#        for p in q:
#         if p.grad is not None:
#          p.grad.fill_(0)
       totalQuality = 0.0

       if True:
           
           inputTensor = numeric # so it will be sequence_length x args.batchSizeHere
#           print inputTensor
#           quit()

           inputTensorIn = inputTensor[:-1]
           inputTensorOut = inputTensor[1:]

           inputEmbeddings = word_pos_morph_embeddings(inputTensorIn.view(args.horizon, batchSizeHere))
           if doDropout:
              inputEmbeddings = inputDropout(inputEmbeddings)
              if args.dropout_rate > 0:
                 inputEmbeddings = dropout(inputEmbeddings)




           lossesWordTotal = []

           sampled_vectors = []
           logProbConditionals = []

           for i in range(inputEmbeddings.size()[0]):
  #            if hidden is not None:
   #              print(i, hidden.abs().max())
              output1, hidden = rnn_both(inputEmbeddings[i].unsqueeze(0), hidden)
   
              assert args.rnn_layers == 1
              meanHidden = hidden[0]
   
              klLoss = [None for _ in inputEmbeddings]
              logStandardDeviationHidden = hiddenToLogSDHidden(hidden[0])
   #           print(torch.exp(logStandardDeviationHidden))
              scaleForDist = torch.log(1+torch.exp(logStandardDeviationHidden))
              memoryDistribution = torch.distributions.Normal(loc=meanHidden, scale=scaleForDist)
   #           sampled = memoryDistribution.rsample()
   
              encodedEpsilon = standardNormalPerStep.sample()
              sampled = meanHidden + scaleForDist * encodedEpsilon
   

              sampled_vectors.append(sampled)
              logProbConditional = memoryDistribution.log_prob(sampled).sum(dim=1)  # TODO not clear whether back-prob through sampled?
              logProbConditionals.append(logProbConditional)


              hiddenNew = sampled.unsqueeze(0)
              # this also serves as the output for prediction
              
              hidden = hiddenNew

#              print(hidden.abs().max())

#              output, _ = rnn_both(torch.cat([word_pos_morph_embeddings(torch.cuda.LongTensor([[2 for _ in range(args.batchSizeHere)]])), inputEmbeddings[halfSeqLen+1:]], dim=0), (hiddenNew, cellNew))
 #             output = torch.cat([output1[:halfSeqLen], output], dim=0)
              output = hiddenNew
              if doDropout:
                 output = dropout(output)
              word_logits = decoder(output)
              word_logits = word_logits.view(batchSizeHere, outVocabSize)
              word_softmax = logsoftmax(word_logits)
              lossesWord = lossModuleTest(word_softmax, inputTensorOut[i].view(batchSizeHere))
              lossesWordTotal.append(lossesWord) 

           lossesWord = torch.stack(lossesWordTotal, dim=0)
           lossWords = lossesWord.sum(dim=0).sum(dim=0)
           loss = lossesWord.sum()


           klLoss = 0
           batchSizeInflatedHere = args.batchSize * len(sampled_vectors)

           sampledTotal = torch.stack(sampled_vectors, dim=0).view(batchSizeInflatedHere, -1)
           logProbConditionalsTotal = torch.stack(logProbConditionals, dim=0).view(batchSizeInflatedHere)
#           print(sampledTotal.size(), logProbConditionalsTotal.size())


           #for sampled, logProbConditional in zip(sampled_vectors, logProbConditionals):
           adjustment = []
           epsilon = sampledTotal
           logdet = torch.autograd.Variable(torch.from_numpy(np.zeros(batchSizeInflatedHere).astype('float32')).cuda())
   #        n=1
           context = torch.autograd.Variable(torch.from_numpy(np.zeros((batchSizeInflatedHere,context_dim)).astype('float32')).cuda())
           for flowStep in range( args.flow_length):
             epsilon, logdet, context = flows[flowStep]((epsilon, logdet, context))
             if flowStep +1 < args.flow_length:
                epsilon, logdet, context = torchkit.flows.FlipFlow(1)((epsilon, logdet, context))
   
           plainPriorLogProb = standardNormal.log_prob(epsilon).sum(dim=1) #- (0.5 * torch.sum(sampled * sampled, dim=1))
           logProbMarginal = plainPriorLogProb + logdet
   
   
           klLoss = (logProbConditionalsTotal - logProbMarginal)
   #           print(logProbConditional, logProbMarginal)
   #           print(logStandardDeviationHidden)
   #           klLoss = 0.5 * (-1 - 2 * (logStandardDeviationHidden) + torch.pow(meanHidden, 2) + torch.exp(2*logStandardDeviationHidden))
    #          klLoss = klLoss.sum(1)


           klLossSum = klLoss.sum()
           if counter % 10 == 0:
              klLossMean = klLoss.mean()
              print(args.beta, args.flow_length, klLossMean, lossesWord.mean(), args.beta * klLoss.mean() + lossesWord.mean() )
              if float(klLossMean) != float(klLossMean):
                 print(hidden.abs().max())
                 assert False, "got NA, abort"
           loss = loss + args.beta * klLossSum
#           print lossesWord

           if surprisalTable is not None or printHere:           
             lossesCPU = lossesWord.data.cpu().view((args.horizon), batchSizeHere).numpy()
             if printHere:
                for i in range(0,args.horizon): #range(1,maxLength+1): # don't include i==0
                         j = 0
                         print (i, itos_total[numeric[i+1][j]], lossesCPU[i][j])

             if surprisalTable is not None: 
                if printHere:
                   print surprisalTable
                for j in range(batchSizeHere):
                  for r in range(args.horizon):
                    surprisalTable[r] += lossesCPU[r,j] #.data.cpu().numpy()[0]

           wordNum = (args.horizon-1)*batchSizeHere
                    
       if wordNum == 0:
         print input_words
         print batchOrdered
         return 0,0,0,0,0
       if printHere:
         print loss/wordNum
         print lossWords/wordNum
         print ["CROSS ENTROPY", crossEntropy, exp(crossEntropy)]
         print ("beta", args.beta)
       crossEntropy = 0.99 * crossEntropy + 0.01 * (lossWords/wordNum).data.cpu().numpy()
       totalQuality = loss.data.cpu().numpy() # consists of lossesWord + lossesPOS
       numberOfWords = wordNum
#       probabilities = torch.sigmoid(dhWeights)
#       neg_entropy = torch.sum( probabilities * torch.log(probabilities) + (1-probabilities) * torch.log(1-probabilities))

#       policy_related_loss = lr_policy * (entropy_weight * neg_entropy + policyGradientLoss) # lives on CPU
       loss = loss / batchSizeHere
       return loss, None, None, totalQuality, numberOfWords, klLoss.mean()


parameterList = list(parameters())

def  doBackwardPass(loss, baselineLoss, policy_related_loss):
       global lastDevLoss
       global failedDevRuns

#       penalty = (hiddenToLogSDHidden.weight * hiddenToLogSDHidden.weight).mean()
 #      loss += 1e-4 * penalty
  #     print(penalty)

       loss.backward()
       if printHere:
         print "BACKWARD 3 "+__file__+" "+args.language+" "+str(args.myID)+" "+str(counter)+" "+str(lastDevLoss)+" "+str(failedDevRuns)+"  "+str(args)
         print devLosses
         print lastDevLoss
#       print("MAX NORM", max(p.grad.data.abs().max() for p in parameterList))
       torch.nn.utils.clip_grad_norm(parameterList, args.norm_clip, norm_type='inf')
       optimizer.step()
       for param in parameters():
         if param.grad is None:
           print "WARNING: None gradient"
#           continue
#         param.data.sub_(lr_lm * param.grad.data)



def createStream(corpus):
#    global counter
    global crossEntropy
    global printHere
    global devLosses

    input_indices = [2] # Start of Segment
    wordStartIndices = []
#    sentenceStartIndices = []
    sentCount = 0
    for sentence in corpus:
       sentCount += 1
       #printHere = (sentCount % 10 == 0)
       ordered = sentence

#       sentenceStartIndices.append(len(input_indices))
       for line in ordered:
          wordStartIndices.append(len(input_indices))
          if line not in stoi:
            input_indices.append(1)
          else:
            input_indices.append(stoi[line]+3)
          if len(wordStartIndices) == args.horizon:
             yield input_indices, wordStartIndices
             input_indices = [2] # Start of Segment (makes sure that first word can be predicted from this token)
             wordStartIndices = []






def computeDevLoss():
   global printHere
#   global counter
#   global devSurprisalTable
   devLoss = 0.0
   devWords = 0
#   corpusDev = getNextSentence("valid")
   corpusDev = corpusIteratorWikiWords.dev(args.language)
   stream = prepareDatasetChunks(corpusDev, train=False)

   surprisalTable = [0 for _ in range(args.horizon)]
   devCounter = 0
   devMemory = 0
   while True:
     devCounter += 1
     printHere = (devCounter % 50 == 0)
     try:
       with torch.no_grad():
          _, _, _, newLoss, newWords, devMemoryHere = doForwardPass(next(stream), surprisalTable = surprisalTable, doDropout=False, batchSizeHere=args.batchSize)
     except StopIteration:
        break

     devMemory += devMemoryHere.data.cpu().numpy()
     devLoss += newLoss
     devWords += newWords
     if printHere:
         print "Dev examples "+str(devCounter)
   devSurprisalTableHere = [surp/(devCounter*args.batchSize) for surp in surprisalTable]
   return devLoss/devWords, devSurprisalTableHere, devMemory/devCounter

DEV_PERIOD = 10000
epochCount = 0
while failedDevRuns < 10:
  epochCount += 1
  print "Starting new epoch, permuting corpus"
  corpus = corpusIteratorWikiWords.training(args.language)
#  stream = createStream(corpus)
  stream = prepareDatasetChunks(corpus, train=True)

  while True:
       counter += 1
       printHere = (counter % 50 == 0)


       if counter % DEV_PERIOD == 0:
          hidden = None
          beginning = zeroBeginning

          newDevLoss, devSurprisalTableHere, newDevMemory = computeDevLoss()
          
          hidden = None
          beginning = zeroBeginning
          

#             devLosses.append(
          devLosses.append(newDevLoss)
          devMemories.append(newDevMemory)
          print "New dev loss "+str(newDevLoss)+". previous was: "+str(lastDevLoss)
#          if newDevLoss > 10:
#              print "Abort, training too slow?"
#              devLosses.append(100)

          if lastDevLoss is None or newDevLoss < lastDevLoss:
              devSurprisalTable = devSurprisalTableHere
         


          print(devSurprisalTable[args.horizon/2])
          print(devMemories)
          with open("/u/scr/mhahn/recursive-prd/memory-upper-neural-pos-only_recursive_words/estimates-"+args.language+"_"+__file__+"_model_"+str(args.myID)+"_"+model+".txt", "w") as outFile:
              print >> outFile, str(args)
              print >> outFile, " ".join(map(str,devLosses))
              print >> outFile, " ".join(map(str,devSurprisalTable))
              print >> outFile, " ".join(map(str, devMemories))
              print >> outFile, str(sum([x-y for x, y in zip(devSurprisalTable[:args.horizon/2], devSurprisalTable[args.horizon/2:])]))
          state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in components]}
          torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")



          if lastDevLoss is None or newDevLoss < lastDevLoss:
             lastDevLoss = newDevLoss
             failedDevRuns = 0
          else:
             failedDevRuns += 1
             print "Skip saving, hoping for better model"
             print devLosses
             print "Epoch "+str(epochCount)+" "+str(counter)
             print zip(range(1,args.horizon+1), devSurprisalTable)
             print "MI(Bottleneck, Future) "+str(sum([x-y for x, y in zip(devSurprisalTable[0:args.horizon/2], devSurprisalTable[args.horizon/2:])]))
             print "Memories "+str(devMemories)
             #break

       try:
         loss, baselineLoss, policy_related_loss, _, wordNumInPass, _ = doForwardPass(next(stream), batchSizeHere=args.batchSize)
       except StopIteration:
          break

       if wordNumInPass > 0:
         doBackwardPass(loss, baselineLoss, policy_related_loss)
       else:
         print "No words, skipped backward"
       if printHere:
          print "Epoch "+str(epochCount)+" "+str(counter)
          print zip(range(1,args.horizon+1), devSurprisalTable)
          if devSurprisalTable[0] is not None:
             print "MI(Bottleneck, Future) "+str(sum([x-y for x, y in zip(devSurprisalTable[0:args.horizon/2], devSurprisalTable[args.horizon/2:])]))
             print "Memories "+str(devMemories)



print(devSurprisalTable[int(args.horizon/2)])
print(devMemories)

