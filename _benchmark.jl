using Pkg
Pkg.activate(".")
using HNSW
using BenchmarkTools

dim = 300
num_elements = 500_000
data = [rand(dim) for i=1:num_elements]

#Intialize HNSW struct
hnsw = HierarchicalNSW(data; efConstruction=100, M=16, ef=50)

#Add all data points into the graph
#Optionally pass a subset of the indices in data to partially construct the graph
add_to_graph!(hnsw)
#
#
queries = [rand(dim) for i=1:1]

k = 10
# Find k (approximate) nearest neighbors for each of the queries
@btime idxs, dists = knn_search(hnsw, queries, k)
