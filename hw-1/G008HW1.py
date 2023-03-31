from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
import argparse
import random
import statistics
import time

from collections import defaultdict

def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count


def edge_parser(val: str):
    values = val.split(',')
    return (int(values[0]), int(values[1]))


def map_phase_1(a, b, u, p, c):
    v1 = u[0]
    h1 = ((a*v1+b) % p) % c
    
    v2 = u[1]
    h2 = ((a*v2+b) % p) % c
    
    if h1 == h2:
        return (h1, (v1, v2)) 


def MR_ApproxTCwithNodeColors(edges, C: int):
	p = 8191
	a = random.randint(1, p-1)
	b = random.randint(0, p-1)

	#R1 M
	map = edges.map(lambda x: map_phase_1(a, b, x, p, C))
	filtered = map.filter(lambda x: x is not None)
        
	groups = filtered.groupByKey()

	#R1 R
	estimates = groups.map(lambda x: (0, CountTriangles(x[1])))
        
	#R2 R
	estimate_sum = estimates.reduceByKey(lambda x, y: x + y)
	sum = estimate_sum.take(1)[0][1]
	
	return C*C*sum


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('C', type=int, help='C value used for estimation')
	parser.add_argument('R', type=int, help='')
	parser.add_argument(
		'file', type=str, help='Path to the input file storing the graph')
	args = parser.parse_args()

	C = args.C
	R = args.R
	file = args.file

	# SPARK SETUP
	conf = SparkConf().setAppName('G008HW1')
	sc = SparkContext(conf=conf)

	# Check that the given input file is actually a file
	assert os.path.isfile(file), "File provided is not valid"

	rawData = sc.textFile(file).cache()
	# print(rawData.take(10))

	# Create edges from string RDD
	edges = rawData.map(edge_parser).repartition(C).cache()
	# print(edges.take(10))

	edge_count = edges.count()

	algorithm_1_estimates = list()
	algorithm_1_execution_times = list()

	for index in range(0,R):
		start_time = time.time()
		algorithm_1_estimates.append(MR_ApproxTCwithNodeColors(edges, C))
		algorithm_1_execution_times.append((time.time() - start_time) * 1000)
    
	median_algorithm_1 = int(statistics.median(algorithm_1_estimates))
	execution_time_algorithm_1 = int(statistics.mean(algorithm_1_execution_times))

	print(f"Dataset =  {file}")
	print(f"Number of Edges = {edge_count}")
	print(f"Number of Colors = {C}")
	print(f"Number of Repetitions = {R}")
	print(f"Approximation through node coloring")
	print(f"- Number of triangles (median over {R} runs) = {median_algorithm_1}")
	print(f"- Running time (average over {R} runs) = {execution_time_algorithm_1} ms")
