import time
from datetime import datetime

# Define the path to the leap seconds table (leap-seconds.list) on your system
leap_seconds_file = "/usr/share/zoneinfo/leap-seconds.list"

def read_leap_seconds_table(file_path):
    leap_seconds = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens = line.strip().split()
            if len(tokens) == 2:
                tai_timestamp, leap_seconds_offset = tokens
                leap_seconds[int(tai_timestamp)] = int(leap_seconds_offset)
    return leap_seconds

def unix_to_tai(unix_timestamp, leap_seconds):
    leap_seconds_offset = 0
    for tai_timestamp, offset in leap_seconds.items():
        if unix_timestamp >= tai_timestamp:
            leap_seconds_offset = offset
        else:
            break
    tai_timestamp = unix_timestamp + leap_seconds_offset
    return tai_timestamp

def tai_to_unix(tai_timestamp, leap_seconds):
    for tai, offset in leap_seconds.items():
        if tai_timestamp >= tai:
            unix_timestamp = tai_timestamp - offset
        else:
            break
    return unix_timestamp

if __name__ == "__main__":
    leap_seconds = read_leap_seconds_table(leap_seconds_file)

    # Convert from Unix time to TAI time
    unix_time = time.time()
    tai_time = unix_to_tai(unix_time, leap_seconds)
    print(f"Unix Time: {unix_time}")
    print(f"TAI Time: {tai_time}")

    # Convert from TAI time to Unix time
    tai_timestamp = 1609459200  # Example TAI timestamp (2023-01-01T00:00:00 TAI)
    unix_timestamp = tai_to_unix(tai_timestamp, leap_seconds)
    print(f"TAI Time: {tai_timestamp}")
    print(f"Unix Time: {unix_timestamp}")

    # You can also convert TAI time to a human-readable date
    tai_datetime = datetime.utcfromtimestamp(tai_timestamp)
    print(f"TAI DateTime: {tai_datetime}")
