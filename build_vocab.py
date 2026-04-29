# Build raw vocabulary
char_set = set()
with open("train.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        label = parts[-1].strip()
        for ch in label:
            if ch.strip() != "":
                char_set.add(ch)

# Sort
char_list = sorted(list(char_set))

# Remove unwanted characters
remove_chars = set([
    '॑', '॒', '॓', '॔',    # noise
    'ॻ', 'ॼ', 'ॽ', 'ॾ',   # rare symbols
    'ॐ', '॰', 'ॱ', 'ॲ'    # vedic marks
])

clean_char_list = []
for ch in char_list:
    if ch in remove_chars:
        continue
    clean_char_list.append(ch)

# Print clean vocab
print("\n--- FINAL VOCAB ---")
for i, ch in enumerate(clean_char_list):
    print(i, f"[{ch}]", ord(ch))
print("\nFinal vocab size:", len(clean_char_list))

# Coverage test
test_words = [
    "अनाथों",
    "इज्ज्त",
    "देखना",
    "मृतका",
    "ऊर्ध्वगामी",
    "१२३"
]

print("\n--- VOCAB COVERAGE TEST ---")
for word in test_words:
    for ch in word:
        if ch not in clean_char_list:
            print(f"Missing character: {ch}")