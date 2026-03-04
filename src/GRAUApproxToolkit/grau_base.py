"""

Extend the measured minimum and maximum range of QuantLinear outputs per layer

"""

def extend_min_max_ranges(min_max_stats):
    extended_min_max_ranges = {}
    for name, stats in min_max_stats.items():
        mins = [s[0] for s in stats]
        maxs = [s[1] for s in stats]
        min_val, max_val = min(mins), max(maxs)
        delta = max_val - min_val
        extended_range = (min_val - delta / 2 * 1, max_val + delta / 2 * 1)
        extended_min_max_ranges[name] = extended_range
    return extended_min_max_ranges