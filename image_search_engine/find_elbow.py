# import packages
from sklearn.cluster import MiniBatchKMeans 
import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--features_db", required = True,
    help = "Path to where the features database will be sorted")
ap.add_argument("-p", "--percentage", type = float, default = 0.25,
    help = "Percentage of total features to use when clustering")
args = vars(ap.parse_args())


# open the database and grab the total number of features
db = h5py.File(args["features_db"])

# determine the number of features to sample, generate the indexes of the
# sample, sorting thme in ascending order to speedup access time from the
# HDF5 database
totalFeatures = db["features"].shape[0]
sampleSize = int(np.ceil(args["percentage"] * totalFeatures))
batch_size = sampleSize // 4
idxs = np.random.choice(np.arange(0, totalFeatures), (sampleSize), replace = False)
idxs.sort()
data = []

# loop over the randomly sampled indexes and accumulate the features to
# cluster
for i in idxs:
    data.append(db["features"][i][2:])

wcss = []
K = [_ for _ in range(100, 3100, 500)]

for k in K: 
    #Building and fitting the model 
    kmeanModel = MiniBatchKMeans(n_clusters=k, batch_size=batch_size)
    kmeanModel.fit(data) 
    wcss.append(kmeanModel.inertia_)
    

plt.plot(K, wcss) 
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.savefig('sampleFileName.png')
plt.show()

# close the database
db.close()




