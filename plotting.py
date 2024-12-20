import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

''' 
Plots zipf's law word distribution for top n words. 
Takes input words = [(word, count), ...] e.g. [("like", 10), ("hello", 9), ...]
'''
def plot_word_distribution(words, title, n=50):
    word_distribution = Counter(words).most_common(n)
    plt.bar(range(n), [w for f, w in word_distribution], align='center')
    plt.xticks(range(n), [f for f, w in word_distribution], rotation=90)
    plt.ylabel("Number of occurences")
    plt.xlabel("Word")
    plt.title(f"Top {n} most frequently occuring words in {title} corpus")
    plt.show()
    

''' plot a simple bar graph '''
def plot_bar_graph(x, y, x_label, y_label, title, ax):
    ax.bar(x, y, align='center')
    ax.set_xticks(x)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


''' create graphs with statistics binned by reading age '''
def plot_reading_age_stats(path):
    with open(path, "r") as f:
        content = f.read()

    # unpack reading age statistics
    rows = content.split("\n")
    headers, rows = rows[0].split(","), rows[1:]
    data = np.array([[float(v) for v in r.split(",")] for r in rows])

    # create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  
    axes = axes.flatten()  

    x = range(len(data[:, 0]))

    # plot graphs
    for i in range(len(headers)):
        plot_bar_graph(x, data[:, i], "reading age", headers[i], f"Reading age against {headers[i]}", axes[i])

    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':
    plot_reading_age_stats("reading_age_statistics.txt")
