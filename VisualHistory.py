import pickle
from VisualNN import *
from VisualLoss import *
from parameters import *

specify = False
locate = False

specify_start = 49

name = '2018-10-29 21-10'
with open('./history/' + name + '.NN', 'rb') as fr:
    myVisualNN = pickle.load(fr)
with open('./history/' + name + '.LOSS', 'rb') as fr:
    myVisualLoss = pickle.load(fr)

# reset plot location
myVisualNN.reset_plot_location()
myVisualLoss.reset_plot_location()

# get a local copy
# myVisualNN = VisualNN(obj=myVisualNN)
# myVisualLoss = VisualLoss(obj=myVisualLoss)

if not specify:
    specify_start = 1
    specify_end = myVisualNN.cnt_history
    for i in range(specify_start, specify_end):
        myVisualNN.plot(i)
        myVisualLoss.plot(i)
        if locate:
            tmp = input()
            locate = False
else:
    current_index = specify_start
    sig = 0
    while True:
        myVisualNN.set_visual_delay(0.1)
        if sig == 'n':
            current_index += 1
        if sig == 'b':
            current_index -= 1
        if sig == 'i':
            myVisualNN.set_visual_delay(5)
        if sig == 'v':
            myVisualNN.set_visual_delay(1)
        if sig == 's':
            break
        myVisualLoss.plot(current_index)
        try:
            myVisualNN.plot(current_index)
        except:
            sig = 'b'
            continue
        sig = input()

tmp = input()
