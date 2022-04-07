import matplotlib.pyplot as plt
import json
import utils.config as config
import os

file_name = os.path.join(config.cache_root, "train_freq.json")

with open(file_name, 'r') as f:
    data = json.load(f)

for key in data:
    freq = data[key]
    freq_scores = sorted(list(freq.values()))[::-1]
    print(freq_scores[:10], key, len(freq_scores))
    plt.figure()
    plt.scatter([x for x in range(len(freq_scores))], freq_scores)
    plt.xlabel('Answers')
    plt.ylabel('Frequencies')
    plt.title(key)
    plt.savefig('plots/'+key+'.png')
    plt.close()