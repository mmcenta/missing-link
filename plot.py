import matplotlib.pyplot as plt

rep_sizes = [64, 128, 256, 512, 1024]
base = [0.73842, 0.74870, 0.76409, 0.79465, 0.81705]
doc2vec = [0.64336, 0.73096, 0.77651, 0.81583, 0.82380]
tfidf = [0.77242, 0.76299, 0.77139, 0.81550, 0.83321]


plt.plot(rep_sizes, base, color='b')
plt.plot(rep_sizes, doc2vec, color='r')
plt.plot(rep_sizes, tfidf, color='b')

plt.savefig("test.png")
