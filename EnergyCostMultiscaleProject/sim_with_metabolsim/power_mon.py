import re
import csv
import subprocess
import time
import sys
import os

job_id = 0
try:
    if len(sys.argv) < 2:
        raise ValueError("Provide JobId as an argument")
    job_id = sys.argv[1]

except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)


nodelist = subprocess.run(["/usr/bin/squeue", "-j", job_id, "--noheader", "--format=%R"], capture_output=True, text=True).stdout

print("\nWill monitor {} nodes for Job Id {} \n".format(nodelist, job_id))

regex = r"^(\S+)\s+(\d+\.\d+)\s*W$"
max_interval = 5
data = {}

output_file = "power_usage_" + job_id + ".csv"
if os.path.exists(output_file):
    os.remove(output_file)

while True:
    start = time.time()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    output = subprocess.run(["sudo", "/usr/local/bin/mpower", nodelist], capture_output=True, text=True).stdout

    for line in output.split("\n"):
        match = re.match(regex, line)
        if match:
            node, power = match.groups()
            data[node] = power

    with open(output_file, "a") as csvfile:
        writer = csv.writer(csvfile)

        if not csvfile.tell():
            writer.writerow(["Time"] + list(data.keys()))

        writer.writerow([timestamp] + list(data.values()))

    end = time.time()
    sleep_time = max_interval - (end - start)
    time.sleep(sleep_time if sleep_time > 0 else 0)

    end = time.time()
    print("{} (interval={:.2f})\n{}".format(timestamp, (end - start), list(data.values())))