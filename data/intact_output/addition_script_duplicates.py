import sys

def main(argv):
    o_pairs = [tuple(i.split()) for i in open(argv[1]).readlines()]
    # o_pairs.sort(key=lambda x : x[0])
    original_set = set(o_pairs)
    num_pairs, num_unique = len(o_pairs), len(original_set)
    for i in open(argv[2]).readlines():
        [x, y] = i .split()
        original_set.add((x, y))
    with open(argv[3], 'w') as new_out:
        for (x, y) in original_set:
            new_out.write(f'{x}\t{y}\n')
    print(f'Original Pairs: {num_pairs} Unique Pairs: {num_unique} Duplicates: {(1 - num_unique / num_pairs) * 100:.2f}% Added: {len(original_set) - num_unique} Final Size: {len(original_set)}')

if __name__ == "__main__":
    main(sys.argv)