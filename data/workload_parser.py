import gzip
import re
import pandas as pd

log_file = "/data1/home/tejashrih/CloudAutoScaling/data/NASA_access_log_Jul95.gz"

timestamps = []

with gzip.open(log_file, "rt", errors="ignore") as f:
    for line in f:
        match = re.search(r"\[(.*?)\]", line)
        if match:
            time_str = match.group(1)

            parts = time_str.split(":")
            if len(parts) >= 3:
                hour = parts[1]
                minute = parts[2]
                timestamps.append([hour, minute])

times = ["{}:{}".format(t[0], t[1]) for t in timestamps if len(t) >= 2]

df = pd.DataFrame(times, columns=["time"])
workload = df.value_counts().reset_index(name="requests")

workload.to_csv("workload.csv", index=False)

print("Workload file created")