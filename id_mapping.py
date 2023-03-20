import sys
from Bio import SeqIO

def get_ids(infile):
    """
    Gets all the unique ids from the original pairs tsv
    """
    with open(infile, 'r') as f:
        with open(f'data/mappings/{infile.split(".")[0].split("/")[2]}_ids.tsv', 'w') as out:
            lines = [l.split() for l in f.readlines()]
            ids = set([l[0] for l in lines]).union(set(l[1] for l in lines))
            for i in ids:
                out.write(f'{i}\n')
                
def make_seqs(ids, mapping):
    """
    Takes in the list of ids, and the mapping file and produces the trimmed sequence file and valid ids
    """
    name = ids.split(".")[0].split("/")[2].split('_')[0]
    ids = set(open(ids, 'r').read().splitlines())
    mapping = {p1 : p2 for [p1, p2] in map(lambda x: x.split(), open(mapping, 'r').read().splitlines()[1:])}
    rmapping = {val: key for key, val in mapping.items()}
    
    # generating the trimmed sequence file
    with open(f'data/seqs/{name}.fasta', 'r') as oSeqs:
        with open(f'data/seqs/{name}_trimmed.fasta', 'w') as nSeqs:
            for record in SeqIO.parse(oSeqs, 'fasta'):
                if record.id in rmapping:
                    nSeqs.write(f'>{rmapping[record.id]}\n')
                    nSeqs.write(f'{record.seq}\n')
    with open(f'data/mappings/{name}_trimmed_ids.tsv', 'w') as newIds:
        newIds.writelines([l + '\n' for l in mapping.keys()])


def main(argv):
    """ argv = [id_mapping.py, mode, pairs/ids, mappings, ]"""
    if argv[1] == '1':
        get_ids(argv[2])
    elif argv[1] == '2':
        make_seqs(argv[2], argv[3])
        
if __name__ == "__main__":
    main(sys.argv)