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

    for doc_id, minhash_signature in enumerate(minhash_signatures):
        candidates = lsh.query(minhash_signature)

        for candidate_id in candidates:
            if doc_id != candidate_id:
                jaccard_estimate = estimate_jaccard_similarity(minhash_signatures[doc_id], minhash_signatures[candidate_id])

                if jaccard_estimate >= threshold:
                    actual_jaccard = jaccard_similarity(shingle_sets[doc_id], shingle_sets[candidate_id])

                    if actual_jaccard >= threshold:
                        pairs.add(frozenset([doc_id, candidate_id]))

    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs}

def naive_strategy(shingle_sets, threshold):
    print("Applying Naive strategy...")
    start_time = time.perf_counter()
    pairs = set()
    false_positives = 0

    for i, shingles_a in enumerate(tqdm(shingle_sets, desc="Naive Strategy")):
        for j, shingles_b in enumerate(shingle_sets[i+1:]):  # Start from i+1 to avoid duplicate pairs
            jaccard_sim = jaccard_similarity(shingles_a, shingles_b)

            if jaccard_sim >= threshold:
                pairs.add(frozenset([i, j+i+1]))
            else:
                false_positives += 1

    end_time = time.perf_counter()
    return {"time": end_time - start_time, "pairs": pairs, "false_positives": 0}  # Add false_positives here
# Implement other strategy functions similarly...



strategies = {
    1: naive_strategy,
    2: lsh_strategy,
}


def plot_data(results):
    strategies = list(results.keys())
    times = [results[strategy]["time"] for strategy in strategies]
    pairs = [len(results[strategy]["pairs"]) for strategy in strategies]
    false_positives = [len(results[strategy]["false_positives"]) for strategy in strategies]

    x = list(range(len(strategies)))

    fig, ax1 = plt.subplots()

    ax1.bar(x, times, label="Time", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.set_ylabel("Time (seconds)")

    ax2 = ax1.twinx()
    ax2.plot(x, pairs, "ro-", label="Pairs Found")
    ax2.plot(x, false_positives, "bo-", label="False Positives")
    ax2.set_ylabel("Number of Pairs and False Positives")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Comparison of Near-Duplicate Detection Strategies")
    plt.show()
    plt.savefig("comparison_plot.png")

# At the end of the main part of the script, add the following line:
plot_data(results)

if __name__ == "__main__":
    print("loading")
    read_file()
    print("Choose a Strategy (1 - Naive Strategy, 2 - LSH Strategy)")
    chosen_strategy = int(input())

    if chosen_strategy not in strategies:
        print("Invalid choice.")
    else:
        if chosen_strategy == 1:
            result = strategies[chosen_strategy](shingle_sets, threshold)
        else:
            result = strategies[chosen_strategy](shingle_sets, minhash_signatures, lsh, threshold)


        plot_data(result)

