import json
from matplotlib.pylab import plt

f = open('./plotdata.json')

data = json.load(f)

for key in data:
    plt.plot(data[key][1:], label=key)

plt.legend(loc='upper right')
plt.title('GCNet', fontsize=16, fontweight='bold')
plt.xlabel('Epochs')
plt.show()