import itertools
import random

import numpy as np
from matplotlib import pyplot as plt

dim_x = 10
dim_y = 10
rf = 2  # current fire range
rm = 6  # maximum fire range
tf = 10  # fire evolution frequency
w = 1.3  # w is a weight for I in the papaer
pedestrain_matrix = np.zeros((dim_x, dim_y))
pedestrains = []
sff = np.zeros((dim_x, dim_y))

# define exits
exit_cells = frozenset((
    (dim_x // 2 - 1, dim_y - 1), (dim_x // 2, dim_y - 1),
    (dim_x - 1, dim_y // 2), (dim_x - 1, dim_y // 2 - 1),
    (0, dim_y // 2 - 1), (0, dim_y // 2),
    (dim_x // 2 - 1, 0), (dim_x // 2, 0),
    # (0,4), (0,5)
))
# print(exit_cells)
# print(np.zeros((10, 10), float))

fire_cells = {(4, 4), (4, 5), (5, 4), (5, 5)}


# initialize walls to be 500
def init_walls(exit_cells):
    sff[0, :] = sff[-1, :] = sff[:, -1] = sff[:, 0] = 500
    pedestrain_matrix[0, :] = pedestrain_matrix[-1, :] = pedestrain_matrix[:, -1] = pedestrain_matrix[:, 0] = 500
    # initialize exit
    for e in exit_cells:
        sff[e] = 0.0001
        pedestrain_matrix[e] = 0


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
    global sff
    for e in exit_cells:
        e_neighbor = get_neighbors(e)
        # print(e_neighbor)
        for c in e_neighbor:
            if c not in exit_cells:
                init_sff_rec(c, 1)
    print(sff)
    sff = np.where(sff != 500, 1 / sff, 500)  # reverse S field value


# a recursive function to initialize static floor field
def init_sff_rec(_cell, _value):
    global sff
    sff[_cell] = _value
    neighbors = get_neighbors(_cell, 0)
    diag_neighbors = get_diag_neighbors(_cell)
    for n in neighbors:
        if n not in exit_cells:
            if sff[n] == 0:
                init_sff_rec(n, _value + 1)

            else:
                if sff[n] > _value + 1:
                    init_sff_rec(n, _value + 1)

    for n in diag_neighbors:
        if n not in exit_cells:
            if sff[n] == 0:
                init_sff_rec(n, _value + 1.5)

            else:
                if sff[n] > _value + 1.5:
                    init_sff_rec(n, _value + 1.5)


init_sff(exit_cells)
print(sff)


# update fire
def update_fire():
    '''todo: further add more rules'''
    for i in fire_cells:
        pedestrain_matrix[i] = 499


# fire evolution by tf
def fire_evolution(t):
    global rf, tf, fire_cells

    if t == 0:
        return

    tmp = set()
    if t % tf == 0 and rf != rm:
        rf = rf + 2
        for i in fire_cells:
            neighbors = get_neighbors(i, 1)
            for j in neighbors:
                tmp.add(j)

    fire_cells = fire_cells.union(tmp)
    update_fire()


print(fire_cells)
fire_evolution(1)
print(fire_cells)


# fire_evolution(20)
# print(fire_cells)
# fire_evolution(30)
# print(fire_cells)

class Rectangle:  # [A,B]
    # [C,D]
    def __init__(self, X, Y, W, H):  # (x,y) are left most coordinates, H height, W width of the box

        self.x = X

        self.y = Y

        self.w = W

        self.h = H

        # check range
        co = self.all_coordinates()
        for i in co:
            if i in fire_cells:
                raise Exception("Fire cells included")
            if i[0] == 0 or i[1] == 0:
                raise Exception("Wall cells included")

    def complete(self):
        return self.x, self.y, self.w, self.h

    def range(self):  # return {A,B,C,D}
        return [(self.x, self.y), (self.x + self.w, self.y), (self.x, self.y - self.h),
                (self.x + self.w, self.y - self.h)]

    def all_coordinates(self):  # return all coordinates
        return list(itertools.product(range(self.x, self.x + self.w + 1), range(self.y - self.h, self.y + 1)))


class Pedestrain:
    def __init__(self, coord):
        if coord in fire_cells:
            raise Exception("Fire cells included")
        if coord in fire_cells:
            raise Exception("Fire cells included")
        self.now = coord
        self.x = coord[0]
        self.y = coord[1]
        if self.x == 0 or self.y == 0:
            raise Exception("Wall cells included")
        if pedestrain_matrix[coord] == 999:
            raise Exception("There is already a pedestrain at {}".format(self.now))
        self.last = coord  # last position
        pedestrain_matrix[coord] = 999
        self.I = np.ones((3, 3), dtype=np.float64)

    def step(self):
        pedestrain_matrix[self.now] = 0
        self.last = self.now
        neighbors = get_neighbors(self.now, moore=1)
        dic = {sff[i]: i for i in neighbors}
        next_cell = dic[max(dic.keys())]
        self.now = next_cell
        pedestrain_matrix[self.now] = 999
        self.update_I()

    @staticmethod
    def update_cell():
        for i in pedestrains:
            pedestrain_matrix[i.now] = 999

    def __str__(self):
        return str(self.now)

    def get_I(self):
        print(self.I)

    def update_I(self):
        self.I = self.I = np.ones((3, 3), dtype=np.float64)
        dir = tuple(map(lambda i, j: i - j, self.now, self.last))
        self.I[(dir[0] + 1, -dir[1] + 1)] = w

    # def burned(self):
    #


rec = Rectangle(1, 2, 1, 1)
print(rec.range())
print(rec.all_coordinates())


def generate_pedestrain_rand(num, rectangle):
    pedestrain_cells = random.sample(rectangle.all_coordinates(), num)
    for i in pedestrain_cells:
        pedestrains.append(Pedestrain(i))


def generate_pedestrain(x):  # x is a tuple|Rectangle
    if isinstance(x, Rectangle):
        for i in x.all_coordinates():
            pedestrains.append(Pedestrain(i))
    if isinstance(x, tuple):
        if x in fire_cells:
            raise Exception("Fire cells included")
        if x[0] == 0 or x[1] == 0:
            raise Exception("Wall cells included")
        pedestrains.append(Pedestrain(x))


# generate_pedestrain_rand(1, rec)
# print(sff)
# print(pedestrain_matrix)
# for i in pedestrains:
#     print("pedestrains:")
#     print(i)


# generate_pedestrain((9, 9))
# print(pedestrain_matrix)
#
# generate_pedestrain(rec)
# print(pedestrain_matrix)
# for i in pedestrains:
#     print("pedestrains:")
#     print(i)

print(pedestrain_matrix)
me = Pedestrain((1, 1))
pedestrains.append(me)
print(pedestrain_matrix)
# print(pedestrain_matrix)
# me.get_I()
# me.step()
# print(pedestrain_matrix)
# me.get_I()
# me.step()
# print(pedestrain_matrix)
# me.step()
# print(pedestrain_matrix)
# me.step()
# print(pedestrain_matrix)

import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap,BoundaryNorm
import matplotlib.animation as animation
import copy

def Update(frameNum, img, oldGrid, nC):
    me.step()
    newGrid=pedestrain_matrix
    displayGrid = copy.deepcopy(newGrid)
    img.set_data(displayGrid)
    oldGrid[:, :] = newGrid[:, :]
    return img

def animate():
    light_gray = np.array([220 / 256, 220 / 256, 220 / 256, 1])
    green = np.array([50 / 256, 200 / 256, 50 / 256, 1])
    red = np.array([255 / 256, 90 / 256, 90 / 256, 1])
    coffee = np.array([111/256, 78/256, 55/256, 1])
    newColors = np.zeros((4, 4))
    newColors[0, :] = light_gray
    newColors[1, :] = red
    newColors[2, :] = coffee
    newColors[3, :] = green
    print(newColors)
    Cmap = ListedColormap(newColors)
    boundary_norm = BoundaryNorm([-0.5, 498.5, 499.5, 500.5, 999.5], Cmap.N)


    fig, ax = plt.subplots()
    ax.axis('off')

    oldGrid = pedestrain_matrix

    img = ax.imshow(oldGrid, cmap=Cmap, interpolation='nearest', norm=boundary_norm)
    ani = animation.FuncAnimation(fig, Update, fargs=(img, oldGrid, 10),
                                  frames=128,
                                  interval=200,
                                  repeat=False)
    f = r'C:\Users\shuox\Desktop\test.mp4'
    ani.save(f, writer='ffmpeg', fps=1)

animate()