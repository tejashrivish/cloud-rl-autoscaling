# import pandas as pd

# df = pd.read_csv("data/workload.csv")

# # convert time to datetime
# df["time"] = pd.to_datetime(df["time"], format="%H:%M")

# # sort by time
# df = df.sort_values("time")

# # convert back to HH:MM format
# df["time"] = df["time"].dt.strftime("%H:%M")

# df.to_csv("data/workload_sorted.csv", index=False)

# print("Sorted workload saved as workload_sorted.csv")


import pandas as pd

df = pd.read_csv(
    "/data1/home/tejashrih/CloudAutoScaling/data/NASA_access_log_Jul95.gz",
    sep=r"\s+",
    header=None,
    usecols=[3, 4],
    engine="python",
    encoding="latin-1",
    on_bad_lines="skip"
)

# Combine timestamp
df["timestamp"] = df[3] + " " + df[4]

# Convert to datetime
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="[%d/%b/%Y:%H:%M:%S %z]"
)

# Create workload per minute
workload_series = (
    df.set_index("timestamp")
      .resample("1min")
      .size()
)

# Convert to DataFrame
workload_df = workload_series.reset_index()
workload_df.columns = ["time", "requests"]

# Convert time to HH:MM (optional but cleaner)
workload_df["time"] = workload_df["time"].dt.strftime("%H:%M")

# Save to CSV
workload_df.to_csv("/data1/home/tejashrih/CloudAutoScaling/data/workload_sorted.csv",index=False)

print("✅ Workload saved to data/workload_sorted.csv")
print(workload_df.head())