outputs = [[1, 2, 3], [4, 5, 6]]
color = {"R": (255, 0, 0), "G": (0, 255, 0), "B": (0, 0, 255)}
axes_misc = {"X": [15, 100], "Y": [30, 100], "Z": [45, 1]}

x, y, z = [outputs[0][i] for i in range(len(outputs[0]))]

for i in [x, y, z]:
    for j in axes_misc.keys():
        for k in color.keys():
            print(i, j, k, color[k], axes_misc[j])
