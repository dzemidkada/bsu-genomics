from IPython.core.display import HTML

COLORS_DICT = {
    -1: 'black',
    0: 'red',
    1: 'blue',
    2: 'green'
}

def display_seq(text, colors):
    result = '\n'.join([
        f'<font color="{c}">{ch}</font>'
        for ch, c in zip(text, colors)])
    display(HTML(f'<b>{result}</b>'))
    
def categorize_matches(diffs, pattern_len, max_diffs_limit=0):
    result = [-1] * len(diffs)
    i = 0
    while i < len(diffs):
        if diffs[i] <= max_diffs_limit:
            result[i] = 2
            for j in range(i+1, min(len(diffs), i + pattern_len)):
                result[j] = diffs[i]
            i += pattern_len
            continue
        i += 1
    return result