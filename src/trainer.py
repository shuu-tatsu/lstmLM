import torch
import torch.optim as optim
import lm as L # 作成したlm.py
import dataset # 作成したdataset.py

maxEpoch = 10 # 最大学習回数
dictPath = '../model/ds.pickle'
modelPath = '../model/lm_%d.model'
trainDataPath = '../data/ptb.train.txt'

# Datasetをdsとして定義して辞書を保存
ds = dataset.Dataset(trainDataPath)
ds.save(dictPath)

# LMをlmとして定義
lm = L.LM(len(ds.word2id))
lm.train() # モデルをtrainモードにする

# optim.Adamをoptとして定義
opt = optim.Adam(lm.parameters())

# 学習イテレーション
for ep in range(maxEpoch):
  accLoss = 0 #エポックごとのロス
  for idLine in ds.idData[0:10]:
    opt.zero_grad() # パラメータの微分値を削除

    ### gradの初期化###
    loss = lm.getLoss(idLine) # ロスを計算
    loss.backward() # 微分(誤差逆伝播)
    opt.step() # 更新反映

    ### 逆伝播して更新###
    accLoss += loss.data[0]
  print(accLoss/len(ds.idData))

  # モデルを’../model/lm_%d.model’としてセーブ
  torch.save(lm, '../model/lm_%d.model'%(ep))
print('FINISH TRAINING')

