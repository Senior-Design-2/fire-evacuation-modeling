import numpy as np

dim_x = 10
dim_y = 10
rf = 2 # current fire range
rm = 6 # maximum fire range
tf = 10 # fire evolution frequency

sff = np.zeros((dim_x, dim_y))

# define exits
exit_cells = frozenset((
                        (dim_x // 2-1, dim_y - 1), (dim_x // 2, dim_y - 1),
                        (dim_x - 1, dim_y//2) , (dim_x - 1, dim_y//2-1),
                        (0, dim_y//2 -1) , (0, dim_y//2),
                        (dim_x//2 -1, 0) , (dim_x//2, 0),
    #(0,4), (0,5)
))
# print(exit_cells)
# print(np.zeros((10, 10), float))

fire_cells = {(4, 4), (4,5), (5,4), (5,5)}



# initialize walls to be 500
def init_walls(exit_cells):
    sff[0, :] = sff[-1, :] = sff[:, -1] = sff[:, 0] = 500

    # initialize exit
    for e in exit_cells:
        sff[e] = -1


init_walls(exit_cells)

print(sff)

# get diagonal neighbors of a cell, return a list of cells
def get_diag_neighbors(cell):
    neighbors = []
    i, j = cell
    if i >= 1 and j >= 1 and sff[(i - 1, j - 1)] != 500:
        neighbors.append((i - 1, j - 1))
    if i < dim_x - 1 and j < dim_y - 1 and sff[(i + 1, j + 1)] != 500:
        neighbors.append((i + 1, j + 1))
    if i < dim_x - 1 and j >= 1 and sff[(i + 1, j - 1)] != 500:
        neighbors.append((i + 1, j - 1))
    if i >= 1 and j < dim_y - 1 and sff[(i - 1, j + 1)] != 500:
        neighbors.append((i - 1, j + 1))
    return neighbors

# get neighbors of a cell, default to be von Neumann neighborhood, if second argument = 1, then get moore neighbor
# return a list of cells
def get_neighbors(cell, moore=0):

    # von Neumann neighborhood

    neighbors = []
    i, j = cell
    if i < dim_x - 1 and sff[(i + 1, j)] != 500:
        neighbors.append((i + 1, j))
    if i >= 1 and sff[(i - 1, j)] != 500:
        neighbors.append((i - 1, j))
    if j < dim_y - 1 and sff[(i, j + 1)] != 500:
        neighbors.append((i, j + 1))
    if j >= 1 and sff[(i, j - 1)] != 500:
        neighbors.append((i, j - 1))

    # moore
    if moore:
        if i >= 1 and j >= 1 and sff[(i - 1, j - 1)] != 500:
            neighbors.append((i - 1, j - 1))
        if i < dim_x - 1 and j < dim_y - 1 and sff[(i + 1, j + 1)] != 500:
            neighbors.append((i + 1, j + 1))
        if i < dim_x - 1 and j >= 1 and sff[(i + 1, j - 1)] != 500:
            neighbors.append((i + 1, j - 1))
        if i >= 1 and j < dim_y - 1 and sff[(i - 1, j + 1)] != 500:
            neighbors.append((i - 1, j + 1))

    return neighbors


# initial static floor field
def init_sff(exit_cells):
    for e in exit_cells:
        e_neighbor = get_neighbors(e)
        # print(e_neighbor)
        for c in e_neighbor:
            if c not in exit_cells:
                init_sff_rec(c, 1)


# a recursive function to initialize static floor field
def init_sff_rec(_cell, _value):
    global sff
    sff[_cell] = _value
    neighbors = get_neighbors(_cell, 0)
    diag_neighbors = get_diag_neighbors(_cell)
    for n in neighbors:
        if n not in exit_cells:
            if sff[n] == 0:
                init_sff_rec(n, _value+1)

            else:
                if sff[n] > _value+1:
                    init_sff_rec(n, _value+1)

    for n in diag_neighbors:
        if n not in exit_cells:
            if sff[n] == 0:
                init_sff_rec(n, _value+1.5)

            else:
                if sff[n] > _value+1.5:
                    init_sff_rec(n, _value+1.5)


init_sff(exit_cells)
print(sff)


# update fire
def update_fire():
    '''todo: further add more rules'''
    for i in fire_cells:
        sff[i] = 500


# fire evolution by tf
def fire_evolution(t):
    global rf, tf, fire_cells

    if t == 0:
        return

    tmp = set()
    if t % tf == 0 and rf != rm:
        rf = rf+2
        for i in fire_cells:
            neighbors = get_neighbors(i, 1)
            for j in neighbors:
                tmp.add(j)

    fire_cells = fire_cells.union(tmp)
    update_fire()


print(fire_cells)
fire_evolution(10)
print(fire_cells)
fire_evolution(20)
print(fire_cells)
fire_evolution(30)
print(fire_cells)