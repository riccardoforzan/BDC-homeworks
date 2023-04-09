from pyspark import SparkContext, SparkConf
import os
import argparse
import random
import statistics
import time

from collections import defaultdict


def count_triangles(edges):
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
    """Extracts integers from a string that contains integer numbers comma separated

    Args:
        val (str): source string

    Returns:
        tuple: tuple containing the two integers
    """
    values = val.split(",")
    return (int(values[0]), int(values[1]))


def map_a1r1(a, b, edge, p, c):
    """Function used to perform map phase of round 1 in ALGORITHM 1, uses the given values of a,b,p,c to assign a color to each vertex of the given edge

    Args:
        a (int): random number in interval (1, p-1)
        b (int): random number in interval (0, p-1)
        edge (RDD): RDD of edges
        p (int): prime number
        c (int): number of colors

    Returns:
        If the two assigned colors are equal returns the edge (tuple) with the assigned color, otherwise returns None
    """
    v1 = edge[0]
    h1 = ((a * v1 + b) % p) % c

    v2 = edge[1]
    h2 = ((a * v2 + b) % p) % c

    if h1 == h2:
        return (h1, (v1, v2))
    else:
        return None


def MR_ApproxTCwithNodeColors(edges, C: int):
    """Computes an estimate of the number of triangles given a list of edges using node-coloring

    Args:
        edges (RDD): RDD of edges
        C (int): number of colors

    Returns:
        int: approximate number of triangles
    """
    p = 8191
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)

    # MAP PHASE (R1)
    map = edges.map(lambda x: map_a1r1(a, b, x, p, C))
    filtered = map.filter(lambda x: x is not None)

    # SHUFFLE+GROUPING
    groups = filtered.groupByKey()

    # REDUCE PHASE (R1)
    estimates = groups.map(lambda x: (0, count_triangles(x[1])))

    # REDUCE PHASE (R2)
    estimate_sum = estimates.reduceByKey(lambda x, y: x + y)

    # compute the estimated number of triangles from obtained sum
    sum = estimate_sum.take(1)[0][1]
    return C * C * sum


def map_a2_r1(edges):
    """Performs the MAP phase of round 1 in ALGORITHM 2

    Args:
        edges (RDD): RDD of edges

    Returns:
        list: list containing only 1 element, namely the number of triangles in the RDD
    """
    return [count_triangles(edges)]


def MR_ApproxTCwithSparkPartitions(edges, C: int):
    """Computes an estimate of the number of triangles given a list of edges using spark partitions

    Args:
        edges (RDD): RDD of edges
        C (int): number of colors

    Returns:
        int: approximate number of triangles
    """
    # MAP PHASE
    estimates = edges.mapPartitions(map_a2_r1).cache()

    # REDUCE PHASE
    sum = estimates.reduce(lambda x, y: x + y)

    # compute the estimated number of triangles from obtained sum
    return C * C * sum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("C", type=int, help="C value used for estimation")
    parser.add_argument("R", type=int, help="Number of repetitions used for estimation")
    parser.add_argument(
        "file", type=str, help="Path to the input file storing the graph"
    )
    args = parser.parse_args()

    C = args.C
    R = args.R
    file = args.file

    # SPARK SETUP
    conf = SparkConf().setAppName("G008HW1")
    sc = SparkContext(conf=conf)

    # Check that the given input file is actually a file
    assert os.path.isfile(file), "File provided is not valid"

    # create RDD of string from given file
    raw_data = sc.textFile(file).cache()

    # Create edges from string RDD
    edges = raw_data.map(edge_parser).repartition(C).cache()
    edge_count = edges.count()

    print(f"\nDataset =  {file}")
    print(f"Number of Edges = {edge_count}")
    print(f"Number of Colors = {C}")
    print(f"Number of Repetitions = {R}")

    # ALGORITHM 1

    algorithm_1_estimates = list()
    algorithm_1_execution_times = list()

    # run MR_ApproxTCwithNodeColors R times and compute median estimate and mean run time
    for index in range(0, R):
        start_time = time.time()
        algorithm_1_estimates.append(MR_ApproxTCwithNodeColors(edges, C))
        algorithm_1_execution_times.append((time.time() - start_time) * 1000)

    median_algorithm_1 = int(statistics.median(algorithm_1_estimates))
    execution_time_algorithm_1 = int(statistics.mean(algorithm_1_execution_times))

    # print results of ALGORITHM 1
    print(f"Approximation through node coloring")
    print(f"- Number of triangles (median over {R} runs) = {median_algorithm_1}")
    print(f"- Running time (average over {R} runs) = {execution_time_algorithm_1} ms")

    # ALGORITHM 2

    start_time_algorithm_2 = time.time()
    algorithm_2_estimate = MR_ApproxTCwithSparkPartitions(edges, C)
    algorithm_2_execution_time = int((time.time() - start_time_algorithm_2) * 1000)

    # print results of ALGORITHM 2
    print(f"Approximation through Spark partitions")
    print(f"- Number of triangles = {algorithm_2_estimate}")
    print(f"- Running time = {algorithm_2_execution_time} ms")
