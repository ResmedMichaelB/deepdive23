import matplotlib.pyplot as plt
import numpy as np

def plot_oct(sig,title_=''):
    
    sig=np.flipud(sig)
    plt.figure(figsize=(5,3))
    ticks=[]
    for i in range(sig.shape[0]):
        plt.plot(sig[i,:]+i*2.5)
        ticks.append(i*2.5)
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Channels')
    plt.yticks(ticks,range(sig.shape[0]))
    plt.title(title_)
