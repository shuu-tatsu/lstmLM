import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

embedSize = 100
lstmHid = 200
lstm_depth = 1

class LM(nn.Module):
    def __init__(self, vocSize):
        super(LM,self).__init__()
        self.embed = nn.Embedding(vocSize, embedSize) ## embedding初期化 ##
        self.lstm = nn.LSTM(input_size=embedSize, hidden_size=lstmHid, num_layers=1, dropout=0.5) ## lstm初期化 ##

        self.linear = nn.Linear(lstmHid, vocSize) ## linear初期化 ##
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, idLine):
        ems = self.embed(Variable(torch.LongTensor(idLine))) # idLineは文の単語id列 ## embedding (len(idLine)*embedSize) ##
        hid = (Variable(torch.zeros(lstm_depth, 1, lstmHid)), Variable(torch.zeros(lstm_depth, 1, lstmHid)))

        #LSTM#
        ys, _ = self.lstm(ems.view(len(idLine),1,-1), hid)
        ys = ys.view(len(idLine),-1) # Linearで扱えるように変形
        
        zs = self.linear(ys) ## ysをlinearに通す ##
        zs_log_softmax = F.log_softmax(zs, dim=1)  ## zsをlog_softmaxに通す ##
        return zs_log_softmax

    #getLoss
    def getLoss(self, idLine):
        zs = self.forward(idLine[:-1])
        ts = Variable(torch.LongTensor(idLine[1:]))
        loss = self.criterion(zs, ts)
        return loss

    def getSentenceLogProb(self, idLine):
        inp = idLine[:-1]
        zs = self.forward(inp) # log_softmaxの結果が返ってくる
        # i番目の確率分布における
        # i+1単語に対応する確率を取り，足し合わせる
        H = 0
        for i in range(len(idLine)-1):
            H += -zs[i][idLine[i+1]].data[0]
        return H

if __name__ == '__main__':
    #lm = LM(10)
    #print(lm.getLoss([0,1,2,3,4,5,0]))

    lm = LM(10)
    idLine = [0,1,2,3,0]
    # forwardを試す
    print(lm.forward(idLine))
    # getLossを試す
    print(lm.getLoss(idLine))
