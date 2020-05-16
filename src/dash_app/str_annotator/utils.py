import base64
import io
from collections import Counter

import pandas as pd

from dash_app.str_annotator.config import GENERATOR_PATTERNS_TO_CHECK
from str_finder.auto_pg import CasualPatternsGenerator
from str_finder.repeat_pattern import GreedyRepeatPattern
from str_finder.utils import check_repeat_pattern, retrieve_true_alleles


def retrieve_reads(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    lines = [
        line.strip()
        for line in io.StringIO(decoded.decode('utf-8'))
    ]
    return lines[1::4]


def generate_potential_patterns(most_common_reads):
    patterns = Counter()
    for r, occ in most_common_reads:
        read_patterns = CasualPatternsGenerator(r).patterns
        patterns.update(read_patterns)

    potential_patterns = []
    for pattern, _ in patterns.most_common(GENERATOR_PATTERNS_TO_CHECK):
        pattern_info = pd.DataFrame({'Pattern': [pattern]})

        repeat_pattern = GreedyRepeatPattern(pattern)
        alleles, annotations = check_repeat_pattern(repeat_pattern,
                                                    most_common_reads)
        true_alleles = retrieve_true_alleles(alleles)
        for allele_id, (allele, support) in enumerate(true_alleles):
            pattern_info[f'Allele #{allele_id+1}'] = allele
            pattern_info[f'Allele #{allele_id+1}_annotation'] = annotations[allele]
            pattern_info[f'Allele #{allele_id+1}_support (# reads)'] = support
        potential_patterns.append(pattern_info)

    return pd.concat(potential_patterns, axis=0).reset_index(drop=True)
