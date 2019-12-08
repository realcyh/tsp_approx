import math
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-inst", default="./DATA/Atlanta.tsp", type=str, help="Please enter file name here")
parser.add_argument("-alg", default="Approx", type=str, choices=["Approx"],
                    help="Specify the algorithm.")
parser.add_argument("-time", default=600, type=int, help="Specify the cutoff in seconds here.")
args = parser.parse_args()


def read_TSP(filename):
    # read TSP file and return the adj matrix
    locations = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            tmp = [i for i in lines.split()]
            if tmp[0] == "1":
                while tmp[0] != "EOF":
                    locations.append(tmp[1:])
                    lines = file_to_read.readline()
                    tmp = [i for i in lines.split()]
                break
        locations = np.array(locations, dtype=float)
    adj = np.zeros((locations.shape[0], locations.shape[0]))
    for i in range(locations.shape[0]):
        adj[i][i] = float("inf")

    for i in range(locations.shape[0]):
        for j in range(locations.shape[0]):
            if i == j:
                continue
            adj[i][j] = round(math.sqrt(math.pow(locations[i][0] - locations[j][0], 2) + math.pow(locations[i][1] - locations[j][1], 2)))
    return adj


def MST(adj):
    G = np.array(adj)
    mst = []
    visited = [0]
    while len(visited) < len(G):
        (row, col) = np.unravel_index(G[visited].argmin(), G[visited].shape)
        visited.append(col)
        mst.append((visited[row], col))

        s = [(col, v) for v in visited]
        for (k, v) in s:
            G[k][v] = float('inf')
            G[v][k] = float('inf')

    # duplicate mst
    tsm = [(y, x) for (x, y) in mst]
    mst.extend(tsm)
    return mst


def get_tour(mst, tour, prev):
    if prev not in tour:
        tour.append(prev)
        next = [x[1] for x in mst if x[0] == prev]
        if len(next) > 0:
            for node in next:
                get_tour(mst, tour, node)
        else:
            return


def get_road(adj, tour):
    dist = 0
    total = 0
    for i in range(len(tour)-1):
        dist = adj[tour[i]][tour[i+1]]
        total = total + dist
    dist = adj[tour[-1]][tour[0]]
    total = total + dist
    tour.append(tour[0])
    return tour, total


def write_sol(city, cutoff, quality, sol):
    cutoff = str(cutoff)
    quality = str(quality)
    with open(city + "_Approx_" + cutoff + '.sol', 'w') as f:
        f.write(quality)
        f.write('\n')
        for a in sol:
            f.write(str(a) + ', ')


def write_trace(city, cutoff, time, quality):
    cutoff = str(cutoff)
    time = str(time)
    quality = str(quality)
    with open(city + "_Approx_" + cutoff + '.trace', 'w') as f:
        f.write(time + ', ' + quality)


def Approx():
    args.inst = "./DATA/UMissouri.tsp"
    temp = args.inst.split("/")[2]
    city = temp.split(".")[0]
    cutoff = args.time
    adj = read_TSP(args.inst)
    start = time.time()
    prev = 0
    tour = []
    mst = MST(adj)
    get_tour(mst, tour, prev)
    res = get_road(adj, tour)
    tour = res[0]
    total = res[1]
    end = time.time()
    t = end - start
    t = format(t, ".4f")
    print(total)
    print(t)
    write_sol(city, cutoff, total, tour)
    write_trace(city, cutoff, t, total)


if __name__ == "__main__":
    if args.alg == "Approx":
        Approx()

