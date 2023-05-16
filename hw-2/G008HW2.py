from pyspark import SparkContext, SparkConf
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


def count_triangles_2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    # We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)
    # Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    # Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:
        u, v = edge
        node_colors[u] = ((rand_a * u + rand_b) % p) % num_colors
        node_colors[v] = ((rand_a * v + rand_b) % p) % num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted(
                            (node_colors[u], node_colors[v], node_colors[w])
                        )
                        # If triangle has the right colors, count it.
                        if colors == triangle_colors:
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


def hash(a, b, vertex, p, c):
    return ((a * vertex + b) % p) % c


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
    h1 = hash(a, b, v1, p, c)

    v2 = edge[1]
    h2 = hash(a, b, v2, p, c)

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


def map_a2r1(a, b, edge, p, c):
    c1 = hash(a, b, edge[0], p, c)
    c2 = hash(a, b, edge[1], p, c)

    l = list()

    for i in range(0, c):
        l.append((tuple(sorted((c1, c2, i))), edge))

    return l


def MR_ExactTC(edges, C: int):
    p = 8191
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)

    # R1

    map = edges.flatMap(lambda x: map_a2r1(a, b, x, p, C))

    groups = map.groupByKey()

    counts = groups.map(lambda i: (0, count_triangles_2(i[0], i[1], a, b, p, C)))

    # R2

    final = counts.reduceByKey(lambda x, y: x + y)

    return final.take(1)[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("C", type=int, help="C value used for estimation")
    parser.add_argument("R", type=int, help="Number of repetitions used for estimation")
    parser.add_argument(
        "F", type=int, help="0 to run MR_ApproxTCwithNodeColors, 1 to run MR_ExactTC"
    )
    parser.add_argument(
        "file", type=str, help="Path to the input file storing the graph"
    )
    args = parser.parse_args()

    C = args.C
    R = args.R
    F = args.F
    file = args.file

    # SPARK SETUP
    conf = SparkConf().setAppName("G008HW1")
    conf.set("spark.locality.wait", "0s")
    sc = SparkContext(conf=conf)

    # create RDD of string from given file
    raw_data = sc.textFile(file).cache()

    # Create edges from string RDD
    edges = raw_data.map(edge_parser).repartition(32).cache()
    edge_count = edges.count()

    print(f"\nDataset =  {file}")
    print(f"Number of Edges = {edge_count}")
    print(f"Number of Colors = {C}")
    print(f"Number of Repetitions = {R}")

    if F == 0:
        algorithm_1_estimates = list()
        algorithm_1_execution_times = list()

        for index in range(0, R):
            start_time = time.time()
            algorithm_1_estimates.append(MR_ApproxTCwithNodeColors(edges, C))
            algorithm_1_execution_times.append((time.time() - start_time) * 1000)

        median_algorithm_1 = int(statistics.median(algorithm_1_estimates))
        execution_time_algorithm_1 = int(statistics.mean(algorithm_1_execution_times))

        print(f"Approximation through node coloring")
        print(f"- Number of triangles (median over {R} runs) = {median_algorithm_1}")
        print(
            f"- Running time (average over {R} runs) = {execution_time_algorithm_1} ms"
        )

    if F == 1:
        algorithm_2_exact = None
        algorithm_2_execution_times = list()

        for index in range(0, R):
            start_time = time.time()
            algorithm_2_exact = MR_ExactTC(edges, C)
            algorithm_2_execution_times.append((time.time() - start_time) * 1000)

        execution_time_algorithm_2 = int(statistics.mean(algorithm_2_execution_times))

        print(f"Exact algorithm with node coloring")
        print(f"- Number of triangles = {algorithm_2_exact}")
        print(
            f"- Running time (average over {R} runs) = {execution_time_algorithm_2} ms"
        )
