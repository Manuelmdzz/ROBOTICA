import os


print("Current Working Directory:", os.getcwd())

cwd = os.getcwd()

new_path = os.path.join(cwd, "12_YOLO", "1_Intro", "Videos", "people.mp4")

print("Constructed Path:", new_path)
print("Path exists:", os.path.exists(new_path))
print("Is a file:", os.path.isfile(new_path))
