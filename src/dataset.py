from collections import Counter
import pickle

class Dataset:
  def __init__(self, dataPath=None):
    self.data = []	# [[‘<BOS>’,’this’,’is’,…], …]
    self.idData = []	# [[0,10,21,31,41], [0,20,11,…]…]
    self.id2word = {}	# {10:this, 21:is, …}
    self.word2id = {}	# {this:10, is:21, …}
    if dataPath:
        self.setData(dataPath)	# self.dataを作る
        self.setDict()		# idは単語の頻度順にふる
        self.setIdData()        # id辞書を使ってself.idDataを作る

  def save(self, dictPath):
    pickle.dump((self.id2word, self.word2id), open(dictPath, 'wb'))

  def load(self, dictPath):
    self.id2word, self.word2id = pickle.load(open(dictPath, 'rb'))

  def setData(self,dataPath):
    # パスから生文を読み込む
    # 分割して<BOS><EOS>を追加
    with open(dataPath, 'r') as r:
        for line in r:
            tok_line = []
            tok_line.append('<BOS>')
            tok_line.extend(line.strip().split())
            tok_line.append('<EOS>')
            self.data.append(tok_line)
        #print(self.data)

  def setDict(self):
    # self.dataを用いてid化
    # 頻度順にソートしてidをふる

    counter = Counter()
    for words in self.data:
        counter.update(words)
   
    cnt = 0 
    for word, count in counter.most_common():
        self.id2word[cnt] = word
        self.word2id[word] = cnt
        cnt += 1
    #print(self.word2id)
    #print(self.id2word)

  def setIdData(self):
    # self.word2idをつかって、
    # self.idDataをつくる
    for line in self.data:
        word_id = []
        for word in line:
            word_id.append(self.word2id[word])
        self.idData.append(word_id)

    #print(self.idData)

if __name__ == '__main__':
    dataPass = '/Users/shusuke-t/pytorch-study/lstmLM/data/ptb.train.txt'
    dset = Dataset(dataPass)
