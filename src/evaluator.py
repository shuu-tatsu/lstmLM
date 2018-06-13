import torch
import lm as L
import dataset
from tqdm import tqdm

epoch = 0 # 評価したいモデルのエポック

dictPath = '../model/ds.pickle'
testDataPath = '../data/ptb.test.txt'
modelPath = '../model/lm_%d.model'%epoch

ds = dataset.Dataset()

ds.load('../model/ds.pickle')
ds.setData(testDataPath)
ds.setIdData()
lm = L.LM(len(ds.word2id))
lm = torch.load('../model/lm_%d.model'%epoch)
lm.eval() # 評価モードにする

# 評価
H = 0 # エントロピー
W = 0 # 単語数
for idLine in tqdm(ds.idData):
    H += lm.getSentenceLogProb(idLine) ### idLineの尤度を得る###
    W += len(idLine) - 1 ### BOSを抜いた単語数を足す###
H /= W
print('entropy:', H)
print('PPL:', 2**H)
