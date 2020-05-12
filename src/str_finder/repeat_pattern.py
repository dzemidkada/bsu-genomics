class Match:
    def __init__(self, l, r, ref):
        self._l = l
        self._r = r
        self._ref = ref
        
    def __str__(self):
        return f'({self._l}, {self._r})'


class Item:
    def __init__(self):
        pass
    
    def countable(self):
        raise NotImplementedError()


class Delim(Item):
    def __init__(self, s):
        super().__init__()
        self._is_countable = (s == s.upper())
        self._data = s.upper()
    
    def __str__(self):
        if self._is_countable:
            return self._data
        return self._data.lower()

    def match(self, text, position):
        if position + len(self._data) >= len(text):
            return None
        is_match = text[position:position+len(self._data)] == self._data
        if is_match:
            return [Match(position, position+len(self._data), self)]
        return None
        
    def countable(self):
        return self._is_countable
        
        
class Repeat(Item):
    def __init__(self, s):
        super().__init__()
        assert s.startswith('[') and s.endswith(']n'), 'Doesn\'t look like repeat'
        self._data = s[1:-2]
        
    def __str__(self):
        return f'[{self._data}]'
    
    def match(self, text, position):
        matches = []
        while True:
            if position + len(self._data) >= len(text):
                break
            is_match = text[position:position+len(self._data)] == self._data
            if is_match:
                matches.append(Match(position, position+len(self._data), self))
                position += len(self._data)
                continue
            break
            
        if len(matches) < 2:
            matches = None

        return matches

    def countable(self):
        return 1
        

class Wildcard(Item):
    def __init__(self, s):
        super().__init__()
        assert s[0] == 'N'
        self._length = int(s[1:])
    
    def __str__(self):
        return f'N{self._length}'
    
    def match(self, text, position):
        if position + self._length < len(text):
            return [Match(position, position+self._length, self)]
        return None

    def countable(self):
        # TODO(dzmr): Check.
        return 0

        
class BaseRepeatPattern:
    def __init__(self):
        pass
    
    def match(self, text):
        raise NotImplementedError()


class GreedyRepeatPattern(BaseRepeatPattern):
    def __init__(self, pattern):
        super().__init__()
        self._pattern = pattern
        self.__parse_pattern()
        
    def __parse_pattern(self):
        self._units = []
        for unit in self._pattern.split():
            # Repeat
            if unit.startswith('[') and unit.endswith(']n'):
                self._units.append(Repeat(unit))
                continue
            # Wildcard
            if unit.startswith('N'):
                self._units.append(Wildcard(unit))
                continue
            self._units.append(Delim(unit))
            
    def __str__(self):
        return '\n'.join(str(u) for u in self._units)
    
    def match(self, text):
        # Pick starting position
        best_matches, best_sum, annotation = None, 0, 'No matches'
        for start_position in range(len(text)):
            cur_matches = []
            position = start_position
            for unit in self._units:
                u_matches = unit.match(text, position)
                if u_matches:
                    cur_matches.extend(u_matches)
                    position = u_matches[-1]._r
                else:
                    cur_matches = None
                    break
            if cur_matches:
                cur_sum = sum([m._ref.countable() for m in cur_matches])
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_matches = cur_matches
        if best_matches:
            annotation = Annotator().parse(best_matches)
        return best_matches, best_sum, annotation

    
class Annotator:
    def __init__(self):
        pass
    
    def parse(self, matches):
        result = []
        i = 0
        while i < len(matches):
            match = matches[i]
            if isinstance(match._ref, Wildcard) or isinstance(match._ref, Delim):
                result.append(str(match._ref))
                i += 1
                continue
            j = i
            while j < len(matches):
                if match._ref != matches[j]._ref:
                    break
                j += 1
            result.append(f'{str(match._ref)}{j-i}')
            i = j
        
        return ' '.join(result)