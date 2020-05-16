from collections import defaultdict

import numpy as np

from str_finder.utils import *


class DummyMatchFinder:
    def __init__(self, text, pattern):
        self._n = len(text)
        self._m = len(pattern)
        self._text = text + 'A' * (self._m - 1)
        self._pattern = pattern

    @staticmethod
    def _diff_count(a, b):
        return np.sum([
            int((y != '*') & (x != y))
            for x, y in zip(a, b)
        ])

    def find_fuzzy_matches(self):
        result = [0] * self._n
        for i in range(self._n):
            result[i] = self._diff_count(
                self._text[i:i + self._m], self._pattern)
        return result


class TemplateMatcher:

    class PatternMatch:
        def __init__(self, l, r, p_id):
            self._l = l
            self._r = r
            self._id = p_id

        def __str__(self):
            return f'{self._l}:{self._r}, {self._id}'

    def __init__(self, cfg):
        self._cfg = cfg
        self.__parse_config()

    def __parse_config(self):
        self._n = len(self._cfg['template_structure'])
        self._patterns = defaultdict(dict)
        for i, p_cfg in enumerate(self._cfg['template_structure']):
            self._patterns[i]['pattern'] = p_cfg['pattern']
            self._patterns[i]['len'] = len(p_cfg['pattern'])
            self._patterns[i]['is_delim'] = p_cfg['delim']
            self._patterns[i]['max_errors'] = p_cfg['max_errors']

    def __preprocess(self, text):
        for i in range(self._n):
            match_finder = DummyMatchFinder(
                text, self._patterns[i]['pattern'])
            self._patterns[i]['fuzzy_matches'] = match_finder.find_fuzzy_matches()

    def __search_for_start_positions(self):
        result = []
        for i, x in enumerate(self._patterns[0]['fuzzy_matches']):
            if x <= self._patterns[0]['max_errors']:
                result.append(i)
        return result

    def __check_template_match(self, start_position, text_len, text):
        def __add_match(result_matches, x, current_index):
            result_matches.append(
                TemplateMatcher.PatternMatch(
                    current_index,
                    current_index + x['len'] - 1,
                    cp
                )
            )

        current_index = start_position
        result_matches = []
        for cp in range(self._n):
            x = self._patterns[cp]
            if current_index + x['len'] - 1 >= text_len:
                return False, None
            # At least one match is required
            if x['fuzzy_matches'][current_index] > x['max_errors']:
                return False, None
            __add_match(result_matches, x, current_index)
            current_index += x['len']
            if x['is_delim']:
                continue
            while True:
                if current_index >= text_len or x['fuzzy_matches'][current_index] > x['max_errors']:
                    break
                __add_match(result_matches, x, current_index)
                current_index += x['len']

        return True, result_matches

    def __visualize_matches(self, text, matches):
        full_colors = ['black'] * len(text)

        for cp in range(self._n):
            print(f'{cp}: {self._patterns[cp]["pattern"]}')
            n_repeats = 0
            for m in matches:
                if m._id != cp:
                    continue
                match_ = text[m._l:m._r + 1]
                colors = ['green'] + ['blue'] * (len(match_) - 1)
                for i in range(m._l, m._r + 1):
                    full_colors[i] = 'red'
                display_seq(match_, colors)
                n_repeats += 1
            print(f'{n_repeats} repeats\n\n\n\n')

        display_seq(text, full_colors)

    def match(self, text, mode='greedy'):
        self.__preprocess(text)

        potential_start_positions = self.__search_for_start_positions()
        best_matches = []

        for i in potential_start_positions:
            if mode == 'greedy':
                is_match, matches = self.__check_template_match(
                    i, len(text), text)
            else:
                raise NotImplementedError('Only greedy mode is available :(')
            if is_match:
                if len(best_matches) < len(matches):
                    best_matches = matches

        if len(best_matches) == 0:
            print('No matches found')
        else:
            self.__visualize_matches(text, best_matches)


def visualize_matches(text, pattern, max_errors_limit=0):
    match_finder = DummyMatchFinder(text, pattern)
    fuzzy_matches = match_finder.find_fuzzy_matches()
    bins = categorize_matches(fuzzy_matches, len(pattern), max_errors_limit)
    colors = [COLORS_DICT[x] for x in bins]
    display_seq(text, colors)
