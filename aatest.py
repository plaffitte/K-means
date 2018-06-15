import numpy as np
np.random.seed(1337)
import csv
import argparse
from matplotlib import pyplot as plt

''' This script reads a database from a csv file and performs K-means clustering to assign one of five clusters to each data point in the database.
'''

### INITIALIZE SOME VARIABLES ###
threshold = 0

def main(args):

    centroids, names = initializeCentroids() # Initialize centroids coordinates
    dataPoints = []
    dataPath = args.dataPath
    dataPoints = readData(dataPath, dataPoints) # Read data. One same structure (list) to store x, y and cluster assignment
    previousError = 10**9
    stop = False
    i = 0

    while not stop:

        newError = expectation(centroids, dataPoints) # Expectation step: assign cluster

        if args.plot:
            plot(newError, dataPoints)

        maximization(centroids, dataPoints) # Maximization step: calculate new centroids

        print "---> Iteration " + str(i) + ", Error: " + str(newError)

        if (np.abs(previousError - newError)) <= threshold:
            stop = True
            print "Clustering achieved, final error: " + str(newError)
            finish(newError, dataPoints, names)
            return
        if previousError < newError:
            print "Algorithm Diverging! Stopping at iteration " + str(i)
            finish(newError, dataPoints, names)
            return
        i += 1
        previousError = newError

def readData(path, data):

    with open(path, 'rb') as csvfile:
        dataSet = csv.reader(csvfile, delimiter=' ')
        for row in dataSet:
            x, y = row[0].split(',')
            data.append([float(x), float(y)])
            data[-1].append(0) # Add an element to store cluster id for each datapoint.

        return np.asarray(data)

def expectation(centr, data):

    err = 0
    for d in data:
        dx, dy, _ = d
        dist = np.sqrt((float(dx) - centr[:, 0]) ** 2 + (float(dy) - centr[:, 1]) ** 2)
        d[-1] = np.argmin(dist) # assign the cluster that minimizes dist
        err += np.min(dist)

    return err

def maximization(centr, data):

    for k in range(len(centr)):
        clusterPoints = [d for d in data if d[2]==k]
        xmean = np.mean(np.asarray(clusterPoints)[:, 0])
        ymean = np.mean(np.asarray(clusterPoints)[:, 1])
        centr[k] = [xmean, ymean]

def initializeCentroids():

    centNames = {}
    centNames[0] = 'Adam'
    centNames[1] = 'Bob'
    centNames[2] = 'Charley'
    centNames[3] = 'David'
    centNames[4] = 'Edward'
    centValues = np.array([[-0.357, 0.253],[-0.055, 4.392],[2.674, -0.001],[1.044, -1.251],[-1.495, 0.090]])
    # centValues = np.array([[-1, 1],[0, 2],[1, 0.1],[0.2, 1],[-3, 1]])

    return centValues, centNames

def plot(error, data):
    cluster1 = np.asarray([np.asarray(d) for d in data if d[2]==0])
    cluster2 = np.asarray([np.asarray(d) for d in data if d[2]==1])
    cluster3 = np.asarray([np.asarray(d) for d in data if d[2]==2])
    cluster4 = np.asarray([np.asarray(d) for d in data if d[2]==3])
    cluster5 = np.asarray([np.asarray(d) for d in data if d[2]==4])
    fig, ax = plt.subplots()
    if cluster1.any():
        plt.scatter(cluster1[:, 0], cluster1[:, 1], s=100, c='b')
    if cluster2.any():
        plt.scatter(cluster2[:, 0], cluster2[:, 1], s=100, c='g')
    if cluster3.any():
        plt.scatter(cluster3[:, 0], cluster3[:, 1], s=100, c='r')
    if cluster4.any():
        plt.scatter(cluster4[:, 0], cluster4[:, 1], s=100, c='c')
    if cluster5.any():
        plt.scatter(cluster5[:, 0], cluster5[:, 1], s=100, c='y')
    plt.show()

def finish(error, data, centNames):

    plot(error, data)

    with open('OUTPUT.TXT', 'w') as outFile:
        outFile.write('Error = ' + str(np.round(error, 3)))
        print np.shape(data)
        for index in data[:, 2]:
            name = centNames[index]
            outFile.write(str(name)+"\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Performs k-means clustering on dataset.")
    parser.add_argument("dataPath",
                        type=str,
                        help="Path to dataset to be clustered.")
    parser.add_argument("--plot",
                        dest='plot',
                        action='store_true',
                        help="Plot clusters after each iteration.")
    main(parser.parse_args())
