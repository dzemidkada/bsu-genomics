import glob
from collections import Counter, defaultdict

COMPLEMENTARY_DICT = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
}


def get_complementary_read(x):
    return ''.join([COMPLEMENTARY_DICT[_] for _ in x[::-1]])


def read_fq_file(file):
    lines = []
    with open(file, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines[1::4]


def valid_alphabet(read):
    return len(set(COMPLEMENTARY_DICT).union(set(read))) == 4


def filter_loci_reads(reads, min_length=200):
    return [
        read
        for read in reads
        if len(read) > min_length and valid_alphabet(read)
    ]


def get_most_common_reads(reads, k=30):
    reads = filter_loci_reads(reads)
    return Counter(reads).most_common(k)


def get_most_common_reads_from_path(path):
    reads = read_fq_file(path)
    return get_most_common_reads(reads)


def inspect_storage_dir(root_folder, dir_sep='/'):
    '''
    Expected storage structure:
        root_folder / sample / {locus_name}.fq
    '''
    samples_loci_dict = defaultdict(dict)
    loci_samples_dict = defaultdict(dict)
    for sample_path in glob.glob(f'{root_folder}{dir_sep}*'):
        sample_id = sample_path.split(dir_sep)[-1]
        for read_path in glob.glob(f'{sample_path}{dir_sep}*'):
            locus = read_path.split(dir_sep)[-1].split('.')[0]
            reads = get_most_common_reads_from_path(read_path)
            if reads:
                samples_loci_dict[sample_id][locus] = read_path
                loci_samples_dict[locus][sample_id] = read_path
    return samples_loci_dict, loci_samples_dict


def check_repeat_pattern(repeat_pattern, mc_reads):
    alleles = defaultdict(int)
    annotations = dict()
    for read, occ in mc_reads:
        matches = repeat_pattern.match(read)
        comp_matches = repeat_pattern.match(get_complementary_read(read))
        for match_ in [matches, comp_matches]:
            if match_[1] == 0:
                continue
            alleles[match_[1]] += occ
            annotations[match_[1]] = match_[2]
            break
    return alleles, annotations


def retrieve_true_alleles(alleles_candidates):
    alleles = sorted(list(alleles_candidates.items()), key=lambda y: -y[1])[:2]
    if len(alleles) < 2:
        return alleles
    if alleles[0][1] / alleles[1][1] < 3:
        return alleles
    else:
        return alleles[:1]
