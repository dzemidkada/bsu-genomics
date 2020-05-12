from str_finder.repeat_pattern import *
import numpy as np
import sklearn.cluster
from collections import defaultdict, Counter
from Levenshtein import distance as levenshtein_distance


def levenshtein_distance_matrix(words):
    return -1*np.array([[levenshtein_distance(w1,w2) for w1 in words] for w2 in words])


def get_cluster_sizes(x):
    words = np.asarray(x)
    lev_similarity = levenshtein_distance_matrix(words)

    affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    return affprop.labels_
#     return [
#         len(np.unique(words[np.nonzero(affprop.labels_==cluster_id)]))
#         for cluster_id in np.unique(affprop.labels_)
#     ]


def get_k_motifs(s, k):
    motifs = defaultdict(list)
    for i in range(len(s) - k):
        motifs[s[i:i+k]].append(i)
    return sorted([(x, len(y)) for x, y in motifs.items()], key=lambda x: x[1], reverse=True)


def print_motifs(s):
    for k in range(2, 7):
        print(get_k_motifs(s, k)[:10])
        
        
def get_repeats(s, k, min_occ=3):
    motifs = get_k_motifs(s, k)
    results = []
    for motif, fr in motifs:
        repeat = Repeat(f'[{motif}]n')
        best_matches = []

        for i in range(len(s) - k):
            matches = repeat.match(s, i)
            if matches:
                if len(matches) > len(best_matches):
                    best_matches = matches
        
        if len(best_matches) >= min_occ:
            results.append(motif)
    return results


def shifted(x, y):
    return (y in x+x) and (len(y) == len(x))


def reduce_c_shifts(base_repeats):
    results = []
    for x, y in base_repeats:
        is_shift = sum([shifted(x, xx) for xx in sorted(results)])
        if is_shift:
            continue
        results.append(x)
    return sorted(results)


def get_k_motifs(s, k):
    motifs = defaultdict(list)
    for i in range(len(s) - k):
        motifs[s[i:i+k]].append(i)
    return sorted([(x, len(y)) for x, y in motifs.items()], key=lambda x: x[1], reverse=True)
        
    
def get_repeats(s, k, min_occ=3):
    motifs = get_k_motifs(s, k)
    results = []
    for motif, fr in motifs:
        repeat = Repeat(f'[{motif}]n')
        best_matches = []

        for i in range(len(s) - k):
            matches = repeat.match(s, i)
            if matches:
                if len(matches) > len(best_matches):
                    best_matches = matches
        
        if len(best_matches) >= min_occ:
            results.append(motif)
    return results


class CasualPatternsGenerator:
    REPEAT_LENGTH_RANGE = (3, 7)
    
    def __init__(self, s):
        self._s = s
        self._patterns = defaultdict(list)
        self.__generate_patterns()
        
    def __setup_zero_level_repeats(self):
        base_repeats = Counter()
        for rl in range(*CasualPatternsGenerator.REPEAT_LENGTH_RANGE[::-1], -1):
            results = get_repeats(self._s, rl)
            base_repeats.update(results)
        for repeat in base_repeats.keys():
            self._patterns[0].append(f'[{repeat}]n')
        
    def __generate_patterns(self):
        # Level 0
        self.__setup_zero_level_repeats()
        cur_level = 0
        while True:
            promoted_patterns = []
            # Access performance
            for pattern in self._patterns[cur_level]:
                #print(cur_level, pattern, 'Checking')
                rp = GreedyRepeatPattern(pattern)
                match_stat = rp.match(self._s)
                if match_stat[1] > 0:
                #    print(cur_level, pattern, 'Matched')
                    promoted_patterns.append((pattern, match_stat))
                #else:
                #    print('Failure')
            
            self._patterns[cur_level] = sorted([p for p,_ in promoted_patterns])
            
            if len(promoted_patterns) == 0:
                self._last_level = cur_level - 1
                break

            # Branching
            for pattern, ms in promoted_patterns:
                new_pattern_start = ms[0][-1]._r
                repeat_len = len(pattern.split()[-1][1:-2])
                # Append Repeat
                new_pattern = f'[{self._s[new_pattern_start:new_pattern_start+repeat_len]}]n'
                self._patterns[cur_level+1].append(f'{pattern} {new_pattern}')
                # Append Delim + Repeat
                for delim_len in range(*self.REPEAT_LENGTH_RANGE):
                    delim = self._s[new_pattern_start:new_pattern_start+delim_len]
                    new_pattern = self._s[new_pattern_start+delim_len:new_pattern_start+delim_len+repeat_len]
                    if delim != new_pattern and len(new_pattern) == repeat_len:
                        self._patterns[cur_level+1].append(f'{pattern} {delim} [{new_pattern}]n')
            self._patterns[cur_level+1] = list(set(self._patterns[cur_level+1]))
            cur_level += 1
            
    @property
    def patterns(self):
        unleveled_patterns = []
        try:
            for level in range(self._last_level+1):
                unleveled_patterns.extend(self._patterns[level])
        except AttributeError:
            pass
        return sorted(unleveled_patterns)