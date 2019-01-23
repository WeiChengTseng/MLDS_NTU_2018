import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tqdm
import os
import pickle
try:
    plt.style.use('gadfly')
except:
    pass
try:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
except:
    print('no GPU available')
    pass

def plot(name='mnist', limit=10):
    plt.rcParams["figure.figsize"] = (12,6.75)
    his = pickle.load(open('his.p', 'rb'))
    his1, his2 = his['shallow'], his['deep']
    plt.plot(np.arange(len(his1[:300])), his1[:300], label='shallow', alpha=0.7)
    plt.plot(np.arange(len(his2[:300])), his2[:300], label='deep', alpha=0.7)
    plt.legend()
    # plt.ylim([0, limit])
    plt.xlim(left=10)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('{} of mnist with different layers'.format(name))
    
    plt.savefig('hw1-1/{}.png'.format(name), dpi=600)
    plt.close()
    return

if __name__ == '__main__':
    plot()