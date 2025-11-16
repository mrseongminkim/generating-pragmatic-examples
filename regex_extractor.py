import csv

listener_train_regexes = []
with open('data/programs/small-pretrain/listener-train-specs-suffix-idx.tsv', 'r') as f:
    raws = csv.reader(f, delimiter='\t')
    for row in raws:
        _, _, regex = row
        listener_train_regexes.append(regex)
assert len(listener_train_regexes) == 300_000
assert len(set(listener_train_regexes)) == 100_000
listener_train_regexes = set(listener_train_regexes)

listener_valid_regexes = []
with open('data/programs/small-pretrain/listener-validation-specs-suffix-idx.tsv', 'r') as f:
    raws = csv.reader(f, delimiter='\t')
    for row in raws:
        _, _, regex = row
        listener_valid_regexes.append(regex)
assert len(listener_valid_regexes) == 1_000
assert len(set(listener_valid_regexes)) == 1_000
listener_valid_regexes = set(listener_valid_regexes)

assert listener_train_regexes.isdisjoint(listener_valid_regexes)

speaker_train_regexes = []
with open('data/programs/small-pretrain/speaker-train-specs-prefix-idx.tsv', 'r') as f:
    raws = csv.reader(f, delimiter='\t')
    for row in raws:
        context, _, _ = row
        regex = context.split('<extra_id_124>')[-1].split('<extra_id_123>')[0]
        speaker_train_regexes.append(regex)
speaker_train_regexes = set(speaker_train_regexes)

speaker_valid_regexes = []
with open('data/programs/small-pretrain/speaker-validation-specs-prefix-idx.tsv', 'r') as f:
    raws = csv.reader(f, delimiter='\t')
    for row in raws:
        context, _, _ = row
        regex = context.split('<extra_id_124>')[-1].split('<extra_id_123>')[0]
        speaker_valid_regexes.append(regex)
speaker_valid_regexes = set(speaker_valid_regexes)

assert listener_train_regexes == speaker_train_regexes
assert listener_valid_regexes == speaker_valid_regexes

with open('data/programs/pragmatic-target-programs.txt', 'r') as f:
    regexes = f.readlines()
    regexes = [regex.strip() for regex in regexes]
regexes = set(regexes)

assert regexes.isdisjoint(listener_train_regexes)
assert regexes.isdisjoint(listener_valid_regexes)

train_regexes = regexes.union(listener_train_regexes)
with open('prax-train', 'w') as f:
    for regex in train_regexes:
        f.write(repr(regex) + '\n')
with open('prax-valid', 'w') as f:
    for regex in listener_valid_regexes:
        f.write(repr(regex) + '\n')
