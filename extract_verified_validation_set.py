import json

with open('data/full-dataset-with-verifications.json', 'r') as f:
    data = json.load(f)
assert len(data) == 440

with open('data/verified_split_train_set.json', 'r') as f:
    verified_train_data = json.load(f)
assert len(verified_train_data) == 400

train_regex = set()
for item in verified_train_data:
    train_regex.add(item[0])

count = 0
val_items = []
for item in data:
    for option in data[item]['options']:
        if option['ground_truth']:
            regex = option['regex']
            break
    spec = []
    for example in data[item]['examples']:
        spec.append([example['string'], example['label']])
    if regex not in train_regex:
        count += 1
        val_item = [regex, spec]
        val_items.append(val_item)
assert count == 40
with open('data/verified_split_val_set.json', 'w') as f:
    json.dump(val_items, f)
