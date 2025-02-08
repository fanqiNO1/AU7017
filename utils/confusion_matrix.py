import os

import matplotlib.pyplot as plt
import numpy as np

data_string = """
[[ 485  457    6    0    0    1   31    0    0    0]
 [   0 1130    0    0    1    1    2    1    0    0]
 [   0   20 1012    0    0    0    0    0    0    0]
 [   0    6    5  989    0    6    0    1    3    0]
 [   0   40    8    0  934    0    0    0    0    0]
 [   0  133    8    3    1  740    7    0    0    0]
 [  16   80   25    0    4    2  830    0    1    0]
 [   0   59   48    2    2    1    0  916    0    0]
 [   0  112   41    3    2    4   19    7  785    1]
 [   0  459   29    1   12    6    0    4    1  497]]
"""

data = []
for line in data_string.split('\n'):
    line = line.strip()
    if line:
        line = line.replace('[[', '[').replace(']]', ']')
        datas = line.split('[')[1].split(']')[0].split()
        data.append([int(d) for d in datas])

data = np.array(data)

class_names = [str(i) for i in range(10)]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# 设置刻度标记
ax.set(xticks=np.arange(data.shape[1]),
       yticks=np.arange(data.shape[0]),
       xticklabels=class_names,
       yticklabels=class_names,
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

# 在每个单元格中显示数字
thresh = data.max() / 2.
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j,
                i,
                data[i, j],
                ha='center',
                va='center',
                color='white' if data[i, j] > thresh else 'black')

# 调整布局
fig.tight_layout()

this_dir = os.path.dirname(__file__)
plt.savefig(f'{this_dir}/confusion_matrix.png',
            bbox_inches='tight',
            pad_inches=0)
