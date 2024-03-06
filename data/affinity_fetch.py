import sys
import pandas as pd
import requests
from tqdm import trange
import numpy as np
import concurrent.futures
url = "https://digitalgeneai.tech/affinity/predict"
df = pd.read_csv("data/poas_train.csv.gz", compression="gzip")
heavy_sequences = df["HeavyAA_aligned"]
light_sequences = df["LightAA_aligned"]
# make sure no _ is in the sequences
heavy_sequences = [seq.replace("-", "") for seq in heavy_sequences]
light_sequences = [seq.replace("-", "") for seq in light_sequences]
antigen_sequence = "TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEIVKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGAAVNVTVEDERTLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQAVIPSRTVNRKSTDSPVECMG"



# make the post request
def get_data(payload):
    try :
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for 4XX or 5XX status codes
    except Exception as e:
        print(e)
        sys.exit(1)

    result = response.json()
    return result['predicted_affinity']


# for i in trange(10):
#     payload = {
#         "heavy_chain": heavy_sequences[i],
#         "light_chain": light_sequences[i],
#         "antigen": antigen_sequence,
#     }
#     affinity = get_data(payload)
#     print(f"i = {i}, affinity = {affinity}")

# use threading to make the requests
# use 10 threads
import threading
import queue
import time

# predicted_affinities = np.zeros(len(heavy_sequences))
# load the predicted affinities if they exist
try:
    predicted_affinities = np.load("predicted_affinities.npy")
    print("Predicted affinities loaded from file")
except FileNotFoundError:
    predicted_affinities = np.zeros(len(heavy_sequences))
    print("No predicted affinities found, starting from scratch")

# get the index of first 0
first_zero = np.where(predicted_affinities == 0)[0]
# print(f"first zero: {first_zero}")

# Worker function to process sequences
def worker(queue, progress_callback):
    while True:
        i = queue.get()
        if i is None:
            break
        
        payload = {
            "heavy_chain": heavy_sequences[i],
            "light_chain": light_sequences[i],
            "antigen": antigen_sequence,
        }

        print(f"i = {i}")
        affinity = get_data(payload)
        print(f"i = {i}, affinity = {affinity}")
        predicted_affinities[i] = affinity
        progress_callback()
        queue.task_done()

# Function to track progress
def progress_callback():
    global processed_sequences
    processed_sequences += 1
    if processed_sequences % 10 == 0:
        print(f"Progress: {processed_sequences} sequences processed")
        np.save("predicted_affinities.npy", predicted_affinities)

# Number of threads
num_threads = 10



queue = queue.Queue()
for i in first_zero:
    queue.put(i)

# Track the number of processed sequences
processed_sequences = len(predicted_affinities) - len(first_zero)

# Create and start worker threads
threads = []
for _ in range(num_threads):
    thread = threading.Thread(target=worker, args=(queue, progress_callback))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
queue.join()

# Stop workers
for _ in range(num_threads):
    queue.put(None)

# Join threads
for thread in threads:
    thread.join()

