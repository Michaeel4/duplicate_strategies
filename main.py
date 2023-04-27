import xxhash
import time
import os
import pickle
import matplotlib.pyplot as plt
import tqdm
from tqdm.auto import tqdm

file_path = "job-ads.txt"


# The results are used to store the time, pairs and false positives for each strategy
results = {
    "Naive": {"time": 0, "pairs": set(), "false_positives": set()},
    "LSH": {"time": 0, "pairs": set(), "false_positives": set()},
    "MinHash": {"time": 0, "pairs": set(), "false_positives": set()},
    "MinHashFilter": {"time": 0, "pairs": set(), "false_positives": set()},
    "LSHFilter": {"time": 0, "pairs": set(), "false_positives": set()},
}

# Threshold as described in Problem 3
threshold = 0.8

def hash_family(i):
    # Family of hash functions
    # Return a hash function parameterized by i
    def hash_member(x):
        # Convert the integer x to a string and encode it as bytes
        x_bytes = str(x).encode('utf-8')
        return xxhash.xxh32(x_bytes, seed=i).intdigest()
    return hash_member


# creates a shingle by k length
def create_shingles(document, k):
    shingles = set()
    for i in range(len(document) - k + 1):
        shingle = document[i:i+k]
        shingle_hash = hash_family(i)(shingle)
        shingles.add(shingle_hash)
    return shingles



def minwise_hashing(shingles, h):
    signature = []
    for i in range(h):
        hash_func = hash_family(i)
        min_hash = min(hash_func(shingle) for shingle in shingles)
        signature.append(min_hash)
    return signature



class LSH:
    def __init__(self, h, r, b):
        assert r * b == h, "rb should be equal to h"
        self.h = h
        self.r = r
        self.b = b
        self.d = {}

    def insert(self, doc_id, minhash_signature):
        for band in range(self.b):
            band_signature = tuple(minhash_signature[self.r * band:self.r * (band + 1)])
            if band not in self.d:
                self.d[band] = {}
            if band_signature not in self.d[band]:
                self.d[band][band_signature] = set()
            self.d[band][band_signature].add(doc_id)

    def query(self, minhash_signature):
        candidates = set()
        for band in range(self.b):
            band_signature = tuple(minhash_signature[self.r * band:self.r * (band + 1)])
            if band in self.d and band_signature in self.d[band]:
                candidates.update(self.d[band][band_signature])
        return candidates




def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union



def estimate_jaccard_similarity(minhash1, minhash2):
    equal_count = sum(1 for a, b in zip(minhash1, minhash2) if a == b)
    return equal_count / len(minhash1)



announcements = None
shingle_sets = None
minhash_signatures = None


processed_data_path = "processed_data.pkl"

def read_file():
    global announcements
    global shingle_sets
    global minhash_signatures

    if os.path.exists(processed_data_path):
        print("Loading processed data...")
        with open(processed_data_path, "rb") as f:
            data = pickle.load(f)
        announcements = data["announcements"]
        shingle_sets = data["shingle_sets"]
        minhash_signatures = data["minhash_signatures"]
    else:
        print("Processing and saving data...")
        with open(file_path, 'r', encoding='utf-8') as f:
            announcements = [line.strip() for line in tqdm(f.readlines(), desc="Reading Lines")]

        announcements = [announcement.lower() for announcement in announcements]
        shingle_sets = [create_shingles(announcement, k) for announcement in
                        tqdm(announcements, desc="Creating Shingles")]
        minhash_signatures = [minwise_hashing(shingle_set, h) for shingle_set in
                              tqdm(shingle_sets, desc="Minwise Hashing")]

        # Save processed data to a pickle file
        data = {"announcements": announcements, "shingle_sets": shingle_sets, "minhash_signatures": minhash_signatures}
        with open(processed_data_path, "wb") as f:
            pickle.dump(data, f)

    print(announcements[:5])


k = 10
h = 100



# Choose r and b values
r = 5
b = 20

# Create an instance of the LSH class
lsh = LSH(h=h, r=r, b=b)

def lsh_strategy(shingle_sets, minhash_signatures, lsh, threshold):
    print("Applying LSH strategy...")
    start_time = time.perf_counter()
    pairs = set()
    false_positives = set()
    # Insert minhash signatures into the LSH object
    for doc_id, minhash_signature in enumerate(minhash_signatures):
        lsh.insert(doc_id, minhash_signature)

    for doc_id, minhash_signature in enumerate(minhash_signatures):
        candidates = lsh.query(minhash_signature)

        for candidate_id in candidates:
            if doc_id != candidate_id:
                jaccard_estimate = estimate_jaccard_similarity(minhash_signatures[doc_id], minhash_signatures[candidate_id])

                if jaccard_estimate >= threshold:
                    actual_jaccard = jaccard_similarity(shingle_sets[doc_id], shingle_sets[candidate_id])

                    if actual_jaccard >= threshold:
                        pairs.add(frozenset([doc_id, candidate_id]))
                    else:
                        false_positives.add(frozenset([doc_id, candidate_id]))

    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs, "false_positives": false_positives}

def naive_strategy(shingle_sets, threshold):
    print("Applying Naive strategy...")
    start_time = time.perf_counter()
    pairs = set()
    false_positives = set()

    for i, shingles_a in enumerate(tqdm(shingle_sets, desc="Naive Strategy")):
        for j, shingles_b in enumerate(shingle_sets[i+1:]):  # Start from i+1 to avoid duplicate pairs
            jaccard_sim = jaccard_similarity(shingles_a, shingles_b)

            if jaccard_sim >= threshold:
                pairs.add(frozenset([i, j+i+1]))
            else:
                false_positives.add(frozenset([i, j + i + 1]))

    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs, "false_positives": false_positives}
# Implement other strategy functions similarly...
def minhashing_strategy(shingle_sets, minhash_signatures, threshold):
    print("Applying Minhashing strategy...")
    start_time = time.perf_counter()
    pairs = set()
    false_positive = set()

    for i, minhash_a in enumerate(tqdm(minhash_signatures)):
        for j, minhash_b in enumerate(minhash_signatures[i+1:]):
            jaccard_estimate = estimate_jaccard_similarity(minhash_a, minhash_b)

            if jaccard_estimate >= threshold:
                pairs.add(frozenset([i, j+i+1]))
            else:
                false_positive.add(frozenset([i, j+i+1]))


    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs, "false_positives": false_positive}


def minhashing_filtering_strategy(shingle_sets, minhash_signatures, threshold):
    print("Applying Minhashing (Filtering) strategy...")
    start_time = time.perf_counter()
    pairs = set()
    false_positives = set()

    for i, minhash_a in enumerate(tqdm(minhash_signatures)):
        for j, minhash_b in enumerate(minhash_signatures[i+1:]):
            jaccard_estimate = estimate_jaccard_similarity(minhash_a, minhash_b)

            if jaccard_estimate >= threshold:
                actual_jaccard = jaccard_similarity(shingle_sets[i], shingle_sets[j+i+1])

                if actual_jaccard >= threshold:
                    pairs.add(frozenset([i, j+i+1]))
                else:
                    false_positives.add(frozenset([i, j+i+1]))

    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs, "false_positives": false_positives}

import json

def log_results(results, log_file_path):


    print("called me")
    # Convert any sets in the results dictionary to lists
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        else:
            return obj


    results = convert_sets(results)

    print(results)

    with open(log_file_path, "w") as log_file:
        json.dump(results, log_file, indent=4, sort_keys=True)

    for key, value in results.items():
        print(f"{key}: {value}")

def lsh_filtering_strategy(shingle_sets, minhash_signatures, lsh, threshold):
    print("Applying LSH (Filtering) strategy...")
    start_time = time.perf_counter()
    pairs = set()
    false_positives = set()

    for doc_id, minhash_signature in enumerate(tqdm(minhash_signatures)):
        lsh.insert(doc_id, minhash_signature)

    for doc_id, minhash_signature in enumerate(minhash_signatures):
        candidates = lsh.query(minhash_signature)

        for candidate_id in candidates:
            if doc_id != candidate_id:
                actual_jaccard = jaccard_similarity(shingle_sets[doc_id], shingle_sets[candidate_id])

                if actual_jaccard >= threshold:
                    pairs.add(frozenset([doc_id, candidate_id]))
                else:
                    false_positives.add(frozenset([doc_id, candidate_id]))

    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs, "false_positives": false_positives}

def run_all_strategies(strategies, shingle_sets, minhash_signatures, lsh, threshold):


    print("called all strategies")
    results = {}
    for strategy_id, strategy in strategies.items():
        print(f"Running Strategy {strategy_id}")
        if strategy_id == 1:
            result = strategy(shingle_sets, threshold)
        elif strategy_id == 2:
            result = strategy(shingle_sets, minhash_signatures, lsh, threshold)
        elif strategy_id in [3, 4]:
            result = strategy(shingle_sets, minhash_signatures, threshold)
        elif strategy_id == 5:
            result = strategy(shingle_sets, minhash_signatures, lsh, threshold)
        else:
            break

        results[strategy_id] = result

    return results

strategies = {
    1: naive_strategy,
    2: lsh_strategy,
    3: minhashing_strategy,
    4: minhashing_filtering_strategy,
    5: lsh_filtering_strategy,
    6: run_all_strategies,
}

def plot_all_data(results):

    strategy_names = [name for name in results]

    times = [result["time"] for result in results.values()]
    pairs = [len(result["pairs"]) for result in results.values()]
    false_positives = [len(result["false_positives"]) for result in results.values()]

    fig, ax1 = plt.subplots()

    ax1.bar(strategy_names, times, label="Time", alpha=0.7)
    ax1.set_ylabel("Time (seconds)")

    ax2 = ax1.twinx()
    ax2.plot(strategy_names, pairs, "ro-", label="Pairs Found")

    # Check if there are any false positives
    if any(false_positives):
        ax2.plot(strategy_names, false_positives, "bo-", label="False Positives")

    ax2.set_ylabel("Number of Pairs and False Positives")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Near-Duplicate Detection")
    plt.xticks(range(len(strategy_names)), strategy_names)  # Update the x-axis with strategy names
    plt.show()
    plt.savefig("all_strategies_plot.png")


def plot_data(result):
    strategy_name = list(results.keys())[chosen_strategy - 1]
    time_taken = result["time"]
    pairs = len(result["pairs"])
    false_positives = len(result["false_positives"])


    print(f"Time: {time_taken}")
    print(f"Pairs:  {pairs}")
    print(f"False Positives {false_positives}")
    fig, ax1 = plt.subplots()

    ax1.bar(strategy_name, time_taken, label="Time", alpha=0.7)
    ax1.set_ylabel("Time (seconds)")

    ax2 = ax1.twinx()
    ax2.plot(strategy_name, pairs, "ro-", label="Pairs Found")
    ax2.plot(strategy_name, false_positives, "bo-", label="False Positives")
    ax2.set_ylabel("Number of Pairs and False Positives")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title(f"{strategy_name} Near-Duplicate Detection")
    plt.show()
    plt.savefig(f"{strategy_name}_plot.png")

# At the end of the main part of the script, add the following line:

if __name__ == "__main__":
    print("loading")
    read_file()
    print("Choose a Strategy (1 - Naive Strategy, 2 - LSH Strategy, 3 - MinHash Strategy, 4 - MinHash Filtering Strategy, 5 - LSH Filtering, 6 - Run All Strategies)")
    chosen_strategy = int(input())

    if chosen_strategy not in strategies:
        print("Invalid choice.")
    else:
        if chosen_strategy in [1, 2, 3, 4, 5]:
            if chosen_strategy == 1:
                result = strategies[chosen_strategy](shingle_sets, threshold)
            elif chosen_strategy == 2:
                result = strategies[chosen_strategy](shingle_sets, minhash_signatures, lsh, threshold)
            elif chosen_strategy == 3:
                result = strategies[chosen_strategy](shingle_sets, minhash_signatures, threshold)
            elif chosen_strategy == 4:
                result = strategies[chosen_strategy](shingle_sets, minhash_signatures, threshold)
            elif chosen_strategy == 5:
                result = strategies[chosen_strategy](shingle_sets, minhash_signatures, lsh, threshold)

            plot_data(result)
            log_results(result, "experiment_log.txt")

        elif chosen_strategy == 6:
            all_results = strategies[chosen_strategy](strategies, shingle_sets, minhash_signatures, lsh, threshold)



            plot_all_data(all_results)

            log_results(all_results, "experiment_log.txt")
            exit()





