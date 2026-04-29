with open("train.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("Number of training samples:", len(lines))