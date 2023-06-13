from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import argparse
import random
from collections import defaultdict
import statistics

# After how many items should we stop?
THRESHOLD = 10000000


def hash(a: int, b: int, item: int, p: int, c: int):
    return ((a * item + b) % p) % c


# Operations to perform after receiving an RDD 'batch' at time
def process_batch(batch, left: int, right: int, D: int, W: int):
    # We are working on the batch at time
    global stream_length, sub_stream_length, sketch_table, exact_count, hash_function_values, sign_function_values
    batch_size = batch.count()

    # If we already have enough points (> THRESHOLD), skip this batch.
    if stream_length[0] >= THRESHOLD:
        return

    stream_length[0] += batch_size

    batch_items = batch.filter(
        lambda e: left <= int(e) and int(e) <= right).collect()
    sub_stream_length[0] += len(batch_items)

    for element in batch_items:
        item = int(element)

        # exact count update
        if item not in exact_count.keys():
            exact_count[item] = 0

        exact_count[item] += 1

        # sketch table update
        for row in range(D):
            p, a, b = hash_function_values[row]
            col = hash(a, b, item, p, W)

            p, a, b = sign_function_values[row]
            sign = -1 if hash(a, b, item, p, 2) == 0 else 1

            sketch_table[row][col] += sign

    if stream_length[0] >= THRESHOLD:
        stopping_condition.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "D", type=int, help="Number of rows of the count sketch")
    parser.add_argument(
        "W", type=int, help="Number of columns of the count sketch")
    parser.add_argument(
        "left", type=int, help="Left endpoint of the interval of interest"
    )
    parser.add_argument(
        "right", type=int, help="Right endpoint of the interval of interest"
    )
    parser.add_argument(
        "K", type=int, help="Number of top of frequent items of interest"
    )
    parser.add_argument("port_exp", type=int,
                        help="Port number of the remote server")
    args = parser.parse_args()

    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("G008HW3")
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    # conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")

    # Here, with the duration you can control how large to make your batches.
    # Beware that the data generator we are using is very fast, so the suggestion
    # is to use batches of less than a second, otherwise you might exhaust the memory.
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)  # Batch duration of 1 second
    ssc.sparkContext.setLogLevel("ERROR")

    # TECHNICAL DETAIL:
    # The streaming spark context and our code and the tasks that are spawned all
    # work concurrently. To ensure a clean shut down we use this semaphore.
    # The main thread will first acquire the only permit available and then try
    # to acquire another one right after spinning up the streaming computation.
    # The second tentative at acquiring the semaphore will make the main thread
    # wait on the call. Then, in the `foreachRDD` call, when the stopping condition
    # is met we release the semaphore, basically giving "green light" to the main
    # thread to shut down the computation.
    # We cannot call `ssc.stop()` directly in `foreachRDD` because it might lead
    # to deadlocks.
    stopping_condition = threading.Event()

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    print("Receiving data from port =", args.port_exp)

    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    stream_length = [0]  # Stream length (an array to be passed by reference)

    sub_stream_length = [0]  # length of ΣR filtered stream
    sketch_table = [[0] * args.W] * args.D
    exact_count = defaultdict(int)

    hash_function_values = list()
    for _ in range(args.D):
        p = 8191
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)
        hash_function_values.append((p, a, b))

    sign_function_values = list()
    for _ in range(args.D):
        p = 8191
        a = random.randint(1, p - 1)
        b = random.randint(0, p - 1)
        sign_function_values.append((p, a, b))

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream(
        "algo.dei.unipd.it", args.port_exp, StorageLevel.MEMORY_AND_DISK
    )

    # For each batch, to the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(
        lambda batch: process_batch(
            batch, args.left, args.right, args.D, args.W
        )
    )

    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()

    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")

    # NOTE: You will see some data being processed even after the
    # shutdown command has been issued: This is because we are asking
    # to stop "gracefully", meaning that any outstanding work
    # will be done.
    ssc.stop(False, True)
    print("Streaming engine stopped")

    # TRUE SECOND MOMENT
    true_second_moment = 0
    for value in exact_count.values():
        true_second_moment += value**2
    true_second_moment /= sub_stream_length[0] ** 2

    # APPROXIMATE SECOND MOMENT
    approximate_second_moments = list()
    for row_index in range(args.D):
        approximate_second_moments.append(
            sum(item**2 for item in sketch_table[row_index])
        )

    approximate_second_moment = statistics.median(approximate_second_moments)
    approximate_second_moment /= sub_stream_length[0] ** 2

    # TOP K FREQUENCIES
    k_th = sorted([e for e in exact_count.values()], reverse=True)[args.K]

    top_k_th_elements = list()
    for key, freq in exact_count.items():
        if freq >= k_th:
            top_k_th_elements.append(key)

    # FREQUENCY ERROR ESTIMATION
    relative_errors = list()
    top_k_elements = list()
    for u in top_k_th_elements:
        estimate_freq_list = list()

        for row in range(args.D):
            p, a, b = hash_function_values[row]
            col = hash(a, b, u, p, args.W)

            p, a, b = sign_function_values[row]
            sign = -1 if hash(a, b, u, p, 2) == 0 else 1

            estimate_freq_list.append(sign * sketch_table[row][col])

        estimate = statistics.median(estimate_freq_list)
        true_freq = exact_count[u]

        top_k_elements.append((u, true_freq, estimate))

        relative_err = abs(estimate - true_freq) / true_freq
        relative_errors.append(relative_err)

    avg_relative_err = statistics.mean(relative_errors)

    distinct_items = len(exact_count.keys())

    print(
        f"D = {args.D} W = {args.W} [left,right] = [{args.left}, {args.right}] K = {args.K} port={args.port_exp}")

    print(f"Total number of items = {stream_length[0]}")
    print(
        f"Total number of items in [{args.left}, {args.right}] = {sub_stream_length[0]}")
    print(
        f"Number of distinct items in [{args.left}, {args.right}] = {distinct_items}")

    if args.K < 20:
        for item, true_freq, freq in top_k_elements:
            print(f"Item {item} Freq = {true_freq} Est. Freq = {freq}")

    print(f"Avg err for top {args.K} = {avg_relative_err}")

    print(f"F2 {true_second_moment} F2 Estimate {approximate_second_moment}")
