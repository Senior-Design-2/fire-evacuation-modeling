import itertools
import math
from math import sqrt

import numpy
import numpy as np

dim_x = 10
dim_y = 10
rf = 2  # current fire range
rm = 6  # maximum fire range
tf = 10  # fire evolution frequency
w = 1.3  # w is a weight for I in the papaer
ks = 1.0
kd = 1.0
kf = 1.0
time = 1
alpha = 0.2 # coeffision for diffusion
delta = 0.2 # coeefision for decay

myLambda = 0.50  # Sedation probability increase coefficient
pedestrain_matrix = np.zeros((dim_x, dim_y))
pedestrains = []
sff = np.zeros((dim_x, dim_y))
dff = np.zeros((dim_x, dim_y)) # dynamic floor field
dff_diff = np.zeros((dim_x, dim_y)) # dff difference matrix, update current grid then add to dff through update_dff function

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


def get_neighbors_ignore_wall(cell, moore=0):
    # von Neumann neighborhood
    neighbors = []
    i, j = cell
    if i < dim_x - 1:
        neighbors.append((i + 1, j))
    if i >= 1:
        neighbors.append((i - 1, j))
    if j < dim_y - 1:
        neighbors.append((i, j + 1))
    if j >= 1:
        neighbors.append((i, j - 1))
    # moore
    if moore:
        if i >= 1 and j >= 1:
            neighbors.append((i - 1, j - 1))
        if i < dim_x - 1:
            neighbors.append((i + 1, j + 1))
        if i < dim_x - 1:
            neighbors.append((i + 1, j - 1))
        if i >= 1 and j < dim_y - 1:
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
    sff = np.where(sff != 500, 1 / sff, 0)  # reverse S field value
    print(sff)


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
                init_sff_rec(n, _value + sqrt(2))

            else:
                if sff[n] > _value + sqrt(2):
                    init_sff_rec(n, _value + sqrt(2))

'''
init_sff(exit_cells)
print(sff)
'''


def init_dff_diff():
    global dff_diff
    dff_diff = np.zeros((dim_x, dim_y))


def update_dff():
    global dff, dff_diff
    dff += dff_diff

    # iter through all cells in the grid
    for i, j in itertools.chain(itertools.product(range(1, dim_x - 1), range(1, dim_y - 1)), exit_cells):
        for _ in range(int(dff[i, j])):
            if np.random.rand() < delta: # decay
                dff[i, j] -= 1
            elif np.random.rand() < alpha: # diffusion
                dff[i, j] -= 1
                dff[random.choice(get_neighbors((i, j)))] += 1


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

'''
print(fire_cells)
fire_evolution(1)
print(fire_cells)
'''

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
        self.update_xy()
        if self.x == 0 or self.y == 0:
            raise Exception("Wall cells included")
        if pedestrain_matrix[coord] == 999:
            raise Exception("There is already a pedestrain at {}".format(self.now))
        self.last = coord  # last position
        pedestrain_matrix[coord] = 999

    def step(self):
        if self.now in exit_cells:
            print("Exited successfully")
        else:
            self.update()
            pedestrain_matrix[self.now] = 0
            if self.last != self.now:
                dff_diff[self.last] += 1

            self.last = self.now
            max = np.max(self.P)
            max_index = np.where(self.P == max)
            dir = (max_index[0][0]-1, max_index[1][0]-1)
            self.now  = tuple(self.now+np.array(dir))
            self.update_xy()
            pedestrain_matrix[self.now] = 999

    def update(self):
        self.update_I()
        self.update_F()
        self.update_epsilon()
        self.update_n()
        self.update_H()
        self.update_Pc()
        self.update_P()

    def update_P(self):  # overall probability
        print(self.get_S(), "\n I:",self.I, "\n n: ",self.n,"\n epsilon:", self.epsilon, "\n F:",self.F)
        self.P = (np.exp(ks * self.get_S()) * np.exp(kd* self.get_D())*self.I*(1-self.n)*self.epsilon)/np.exp(kf*self.F)
        print("P: ", self.P)

    @staticmethod
    def update_cell():
        for i in pedestrains:
            pedestrain_matrix[i.now] = 999

    def __str__(self):
        return str(self.now)

    def update_xy(self):
        self.x = self.now[0]
        self.y = self.now[1]

    def get_I(self):
        print("I:", self.I)
        return self.I

    def update_I(self): # I -> inertia filed
        self.I = np.ones((3, 3), dtype=np.float64)
        if self.now == self.last:
            return self.I
        dir = tuple(map(lambda i, j: i - j, self.now, self.last))
        self.I[(dir[0] + 1, dir[1] + 1)] = w
        neighbors = self.neighbors((dir[0] + 1, dir[1] + 1))
        for i in neighbors:  # compute the component of inertia on the same direction
            self.I[i] = 1 + 0.8509 * (w - 1)  # sin45 = 0.8509

    def neighbors(self, coord):
        neighbors = []
        for i in [(coord[0] - 1, coord[1]), (coord[0], coord[1] - 1), (coord[0] + 1, coord[1]),
                  (coord[0], coord[1] + 1)]:
            if i != (1, 1) and i[0] <= 2 and i[1] <= 2 and i[0] >= 0 and i[1] >= 0:
                neighbors.append(i)
        return neighbors

    def update_F(self):  # fire floor field, 0 means out of range
        self.F = np.zeros((3, 3))
        neighbors = get_neighbors_ignore_wall(self.now, 1)
        x = self.x - 1
        y = self.y - 1
        neighbors.append(self.now)
        for i in neighbors:
            self.F[i[0] - x, i[1] - y] = self.compute_H(i)
        self.F = 1/self.F
        sum = 0
        for i in self.F:
            sum += i
        self.F = self.F / sum

    def get_F(self):
        print(self.F)
        return self.F

    def compute_H(self, coord):
        temp = 999999
        for i in fire_cells:
            euclidean_distance = np.linalg.norm(np.asarray(coord) - np.asarray(i))
            if temp > euclidean_distance:
                temp = euclidean_distance
        return temp

    def update_H(self):
        temp = 999999
        for i in fire_cells:
            euclidean_distance = np.linalg.norm(np.asarray(self.now) - np.asarray(i))
            if temp > euclidean_distance:
                temp = euclidean_distance
                fire_cell = i
                self.closet_fire_cell = fire_cell
        self.H = temp

    def get_H(self):
        print("H:", self.H)
        return self.H

    def get_closet_fire(self):
        return self.closet_fire_cell

    def update_Pc(self):  # Ｐｃ ＝１－ｅｘｐ（－λ（Ｈ －１））
        self.Pc = 1 - math.exp(-myLambda * (self.H - 1))

    def get_Pc(self):
        print("Pc：", self.Pc)
        return self.Pc

    def update_epsilon(self):  # update obstacle matrix,  epsilon is [] when pedestrian on the border/exit
        temp = pedestrain_matrix[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        self.epsilon = np.where(temp == 500, 0, 1)
        if self.epsilon.size == 0: # epsilon is [] when pedestrian on the border/exit
            self.epsilon = numpy.zeros((3, 3))

    def get_epsilon(self):
        print("Epsilon:", self.epsilon, self.n)
        return self.epsilon

    def update_n(self):  # update target cell occupation matrix, n is [] when pedestrian on the border/exit
        temp = pedestrain_matrix[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        # print(temp)
        self.n = np.where(temp == 999, 1, 0)
        if self.n.size != 0: # n is [] when pedestrian on the border/exit
            self.n[1, 1] = 0
        else:
            self.n = numpy.ones((3, 3))

    def get_n(self):
        return self.n

    def get_S(self):
        s = sff[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        if s.size == 0:
            return numpy.ones((3, 3))
        else:
            return s

    def get_D(self):
        d = dff[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        return d


    # def burned(self):
    #



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

rec = Rectangle(1, 2, 1, 1)
print(rec.all_coordinates())


# generate_pedestrain((9, 9))
# print(pedestrain_matrix)
#

generate_pedestrain(rec)

while(1):
    for i in pedestrains:
        i.step()
    update_dff()
    init_dff_diff()
    print(pedestrain_matrix)




# print(pedestrain_matrix)
# me = Pedestrain((1, 1))
# pedestrains.append(me)
# print(pedestrain_matrix)
# # me.get_I()
# # me.get_H()
# # me.get_Pc()
# # me.get_epsilon()
# me.step()
# # me.get_I()
# # me.get_H()
# # me.get_Pc()
# # me.get_epsilon()
# print(pedestrain_matrix)
#
# me.step()
# # me.get_I()
# # me.get_H()
# # me.get_Pc()
# # me.get_epsilon()
# print(pedestrain_matrix)
#
# me.step()
# # me.get_I()
# # me.get_H()
# # me.get_Pc()
# # me.get_epsilon()
# print(pedestrain_matrix)


import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
import copy


def Update(frameNum, img, oldGrid, nC):
    #me.step()
    newGrid = pedestrain_matrix
    displayGrid = copy.deepcopy(newGrid)
    img.set_data(displayGrid)
    oldGrid[:, :] = newGrid[:, :]
    return img


def animate():
    light_gray = np.array([220 / 256, 220 / 256, 220 / 256, 1])
    green = np.array([50 / 256, 200 / 256, 50 / 256, 1])
    red = np.array([255 / 256, 90 / 256, 90 / 256, 1])
    coffee = np.array([111 / 256, 78 / 256, 55 / 256, 1])
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
    f = r'.\test.mp4'
    ani.save(f, writer='ffmpeg', fps=1)
