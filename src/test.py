outputs = [[1, 2, 3], [4, 5, 6]]
color = {"R": (255, 0, 0), "G": (0, 255, 0), "B": (0, 0, 255)}
axes_misc = {"X": [15, 100], "Y": [30, 100], "Z": [45, 1]}

x, y, z = [outputs[0][i] for i in range(len(outputs[0]))]

for i in [x, y, z]:
    for j in axes_misc.keys():
        for k in color.keys():
            print(i, j, k, color[k], axes_misc[j])

def xy_min_max(p, d, m, f):
    return(int(p + (d * m) // 2) if int(p + (d * m) // 2) >= f else 0)

f_shape = [300, 500]

l_coords = [100, 150]
r_coords = [200, 250]
l_shape = r_shape = [1, 3, 50, 50]
L, R = ([l_coords, l_shape], [r_coords, r_shape])
for i in (L, R):
    x = i[0][0]
    y = i[0][1]
    h = i[1][2]
    w = i[1][3]
    xmin = xy_min_max(x, w, -1, 0)
    xmax = xy_min_max(x, w, 1, f_shape[1])
    ymin = xy_min_max(y, h, -1, 0)
    ymax = xy_min_max(y, h, 1, f_shape[0])
    print(xmin, xmax, ymin, ymax)
