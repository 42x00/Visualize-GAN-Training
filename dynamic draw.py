import matplotlib.pyplot as plt
fig,ax=plt.subplots()
y1=[]
for i in range(50):
    y1.append(i)
    ax.cla()
    ax.bar(y1,label='test',height=y1,width=0.3)
    ax.legend()
    plt.pause(0.3)