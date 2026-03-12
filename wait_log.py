import time
import sys

def tail_file(filepath):
    with open(filepath, 'r') as f:
        # Seek to the end of the file
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            if "STATISTICAL TESTS" in line or "RESULTS SUMMARY" in line:
                # Read the rest of the file
                time.sleep(2)
                for _ in range(50):
                    sys.stdout.write(f.readline())
                break

if __name__ == "__main__":
    tail_file('run_full2.log')
