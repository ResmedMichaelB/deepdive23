import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
from scipy.signal import welch

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

def plot_spectrum(sig):

    f,y=welch(sig)
    plt.plot(f,np.log(y**2))


def train_val_curve(train):
    
    fig=plt.figure()
    accuracy=train.history['accuracy']
    val_accuracy=train.history['val_accuracy']

    loss=train.history['loss']
    val_loss=train.history['val_loss']
    epoch_range = range(len(accuracy))
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(epoch_range, accuracy, label='Training accuracy')
    plt.plot(epoch_range, val_accuracy, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epoch_range, loss, label='Training loss')
    plt.plot(epoch_range, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    return fig


def plot_roc(y_true,y_prob, thresh,label=''):

    fpr,tpr,_=roc_curve(y_true,y_prob)
    plt.plot(fpr,tpr,label=label  + ': ' + str(round(roc_auc_score(y_true,y_prob),2)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()


def plot_binary_prob(y_test, y_pred, name, thresh):

    fig, ax1 = plt.subplots(figsize=(5,2.5))
    ax2 = ax1.twinx()
    ax1.plot(y_test, 'black')
    ax2.plot(y_pred, 'grey')

    ax1.set_xlabel('Time (30s windows)')
    # ax1.set_ylabel('Position', color='black')
    ax1.set_yticks([0,1],['Side','Supine'],rotation=45)
    ax1.set_ylim([-.1,1.1])
    ax2.set_ylabel('Supine Probability', color='grey')
    ax2.axhline(0.7,ls='--',color='tomato')
    ax2.set_ylim([-.1,1.1])
    plt.title('subject {} overnight Accuracy {:.2f}'.format(name, f1_score(y_test,y_pred>thresh)))
    plt.tight_layout()
