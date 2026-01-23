import itertools as it
from os import makedirs
import subprocess

from tqdm import tqdm

# Read filenames
allFiles = []
print("Reading filenames")
with open("failures.txt", 'r') as file:
    allFiles = eval(file.read().strip())

allFiles = list(it.chain.from_iterable(allFiles))

# Download one of the filename lists
def download(name, files_to_download: list, batch_size = 1):
    makedirs(name, exist_ok=True)
    
    prefix = f"./{name}"
    command = ["wget", "-P", prefix]
    failures = []

    # Download files in batches of batch_size
    batches = [list(x) for x in it.batched(files_to_download, batch_size)]
    print(f"Starting downloads for {name}")
    for filenames in tqdm(batches):
        try:
            subprocess.run(command + filenames, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            failures.append(filenames)
            print(f"Failed batch {failures[-1]}.")
            print()

    print("Done. Printing summary of failures:")
    print(failures)
    print("Exporting failure summary.")
    with open("failures.txt", 'w') as file:
        file.write(str(failures))

if __name__ == "__main__":
    download("exports", allFiles)
    # print(allFiles)