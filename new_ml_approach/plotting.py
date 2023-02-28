import matplotlib.pyplot as plt

def plot_oct(sig):
    plt.figure()
    for i in range(sig.shape[0]):
        plt.plot(sig[i,:]+i*2.5)