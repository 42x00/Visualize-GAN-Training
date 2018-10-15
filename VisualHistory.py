import pickle

name = '2018-10-15 20:39'
with open('./history/' + name + '.NN', 'rb') as fr:
    myVisualNN = pickle.load(fr)
with open('./history/' + name + '.LOSS', 'rb') as fr:
    myVisualLoss = pickle.load(fr)

for i in range(myVisualNN.cnt_history):
    myVisualNN.plot(i)
    myVisualLoss.plot(i)
