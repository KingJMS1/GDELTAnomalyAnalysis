import itertools as it
from os import makedirs
import subprocess

from tqdm import tqdm

# Read filenames
exportFiles = []
mentionsFiles = []
gkgFiles = []

allFiles = []
print("Reading filenames")
with open("masterfilelist.txt", 'r') as file:
    allFiles = [x.strip().split()[-1] for x in file.readlines()]

for filename in allFiles:
    if ".mentions" in filename:
        mentionsFiles.append(filename)
    elif ".export" in filename:
        exportFiles.append(filename)
    elif ".gkg" in filename:
        gkgFiles.append(filename)
print("Done reading filenames")


# Download one of the filename lists
def download(name, files_to_download: list, batch_size = 10):
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
    download("exports", exportFiles)
