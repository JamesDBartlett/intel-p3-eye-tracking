from itertools import islice

color = {"R": (255, 0, 0), "G": (0, 255, 0), "B": (0, 0, 255), "C": (0, 255, 255), "M": (255, 0, 255), "Y": (255, 255, 0), "K": (0, 0, 0)}
axes_misc = {"X": [25, 100], "Y": [50, 100], "Z": [75, 1]}

def take(n, iterable):
    return list(islice(iterable, n))

def rgb(x):
    return color[x]

def bgr(x):
    return color[x][::-1]