import numpy as np

dim_x = 10
dim_y = 10

grid = np.zeros((dim_x, dim_y))

# define exits
exit_cells = frozenset(((dim_x // 2-1, dim_y - 1), (dim_x // 2, dim_y - 1),
                        (dim_x - 1, dim_y//2) , (dim_x - 1, dim_y//2-1),
                        (0, dim_y//2 -1) , (0, dim_y//2),
                        (dim_x//2 -1, 0) , (dim_x//2, 0),
    #(0,4), (0,5)
))
print(np.zeros((10, 10), float))


# initialize walls to be 500
def init_walls(exit_cells):
    grid[0, :] = grid[-1, :] = grid[:, -1] = grid[:, 0] = 500

    # initialize exit
    for e in exit_cells:
        grid[e] = -1


init_walls(exit_cells)

print(grid)

# get diagonal neighbors of a cell, return a list of cells
def get_diag_neighbors(cell):
    neighbors = []
    i, j = cell
    if i >= 1 and j >= 1 and grid[(i - 1, j - 1)] != 500:
        neighbors.append((i - 1, j - 1))
    if i < dim_x - 1 and j < dim_y - 1 and grid[(i + 1, j + 1)] != 500:
        neighbors.append((i + 1, j + 1))
    if i < dim_x - 1 and j >= 1 and grid[(i + 1, j - 1)] != 500:
        neighbors.append((i + 1, j - 1))
    if i >= 1 and j < dim_y - 1 and grid[(i - 1, j + 1)] != 500:
        neighbors.append((i - 1, j + 1))
    return neighbors

# get neighbors of a cell, default to be von Neumann neighborhood, if second argument = 1, then get moore neighbor
# return a list of cells
def get_neighbors(cell, moore=0):

    # von Neumann neighborhood

    neighbors = []
    i, j = cell
    if i < dim_x - 1 and grid[(i + 1, j)] != 500:
        neighbors.append((i + 1, j))
    if i >= 1 and grid[(i - 1, j)] != 500:
        neighbors.append((i - 1, j))
    if j < dim_y - 1 and grid[(i, j + 1)] != 500:
        neighbors.append((i, j + 1))
    if j >= 1 and grid[(i, j - 1)] != 500:
        neighbors.append((i, j - 1))

    # moore
    if moore:
        if i >= 1 and j >= 1 and grid[(i - 1, j - 1)] != 500:
            neighbors.append((i - 1, j - 1))
        if i < dim_x - 1 and j < dim_y - 1 and grid[(i + 1, j + 1)] != 500:
            neighbors.append((i + 1, j + 1))
        if i < dim_x - 1 and j >= 1 and grid[(i + 1, j - 1)] != 500:
            neighbors.append((i + 1, j - 1))
        if i >= 1 and j < dim_y - 1 and grid[(i - 1, j + 1)] != 500:
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
    global grid
    grid[_cell] = _value
    neighbors = get_neighbors(_cell, 0)
    diag_neighbors = get_diag_neighbors(_cell)
    for n in neighbors:
        if n not in exit_cells:
            if grid[n] == 0:
                init_sff_rec(n, _value+1)

            else:
                if grid[n] > _value+1:
                    init_sff_rec(n, _value+1)

    for n in diag_neighbors:
        if n not in exit_cells:
            if grid[n] == 0:
                init_sff_rec(n, _value+1.5)

            else:
                if grid[n] > _value+1.5:
                    init_sff_rec(n, _value+1.5)


init_sff(exit_cells)
print(grid)