import itertools
import math
import numpy
import numpy as np
import random
from math import sqrt

dim_x = 10
dim_y = 100

tf = 10  # fire evolution frequency
w = 1.05  # w is a weight for I in the paper
ks = 2
kd = 0.01
kf = 0.05
alpha = 0.2  # coeffision for diffusion
delta = 0.2  # coeefision for decay
myLambda = 0.50  # Sedation probability increase coefficient

pedestrains = []
visual_field = np.zeros((dim_x, dim_y))
sff = np.zeros((dim_x, dim_y))
dff = np.zeros((dim_x, dim_y))  # dynamic floor field
dff_diff = np.zeros(
    (dim_x, dim_y))  # dff difference matrix, update current grid then add to dff through update_dff function


# initialize walls to be 500
def init_walls(exit_cells):
    sff[0, :] = sff[-1, :] = sff[:, -1] = sff[:, 0] = 500
    visual_field[0, :] = visual_field[-1, :] = visual_field[:, -1] = visual_field[:, 0] = 500
    # initialize exit
    for e in exit_cells:
        sff[e] = 0.1
        visual_field[e] = 0


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
def get_neighbors(cell, moore=0, ignore_val=500):
    # von Neumann neighborhood
    neighbors = []
    i, j = cell
    if i < dim_x - 1 and sff[(i + 1, j)] != ignore_val:
        neighbors.append((i + 1, j))
    if i >= 1 and sff[(i - 1, j)] != ignore_val:
        neighbors.append((i - 1, j))
    if j < dim_y - 1 and sff[(i, j + 1)] != ignore_val:
        neighbors.append((i, j + 1))
    if j >= 1 and sff[(i, j - 1)] != ignore_val:
        neighbors.append((i, j - 1))

    # moore
    if moore:
        if i >= 1 and j >= 1 and sff[(i - 1, j - 1)] != ignore_val:
            neighbors.append((i - 1, j - 1))
        if i < dim_x - 1 and j < dim_y - 1 and sff[(i + 1, j + 1)] != ignore_val:
            neighbors.append((i + 1, j + 1))
        if i < dim_x - 1 and j >= 1 and sff[(i + 1, j - 1)] != ignore_val:
            neighbors.append((i + 1, j - 1))
        if i >= 1 and j < dim_y - 1 and sff[(i - 1, j + 1)] != ignore_val:
            neighbors.append((i - 1, j + 1))
    return neighbors


def get_neighbors_including_wall(cell, moore=0):
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
    sff = np.where(sff != 500, 1 / sff, 0)  # reverse S field value


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


def init_dff_diff():
    global dff_diff
    dff_diff = np.zeros((dim_x, dim_y))


def update_dff():
    global dff, dff_diff
    dff += dff_diff
    # iter through all cells in the grid
    for i, j in itertools.chain(itertools.product(range(1, dim_x - 1), range(1, dim_y - 1)), exit_cells):
        for _ in range(int(dff[i, j])):
            if np.random.rand() < delta:  # decay
                dff[i, j] -= 1
            elif np.random.rand() < alpha:  # diffusion
                dff[i, j] -= 1
                dff[random.choice(get_neighbors((i, j)))] += 1


# update fire
def update_fire():
    '''todo: further add more rules'''
    for i in fire_cells:
        visual_field[i] = 499


# fire evolution by tf
def fire_evolution(t):
    global rf, tf, fire_cells

    if t == 0:
        update_fire()
        return

    tmp = set()
    if t % tf == 0:
        for i in fire_cells:
            neighbors = get_neighbors(i, 1, 0)
            for j in neighbors:
                tmp.add(j)
    fire_cells = fire_cells.union(tmp)
    update_fire()


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
            if i[0] == 0 or i[1] == 0 or i[0] == dim_x-1 or i[1] == dim_y-1:
                raise Exception("Wall cells {} included".format(i))

    def complete(self):
        return self.x, self.y, self.w, self.h

    def range(self):  # return {A,B,C,D}
        return [(self.x, self.y), (self.x + self.w, self.y), (self.x, self.y - self.h),
                (self.x + self.w, self.y - self.h)]

    def all_coordinates(self):  # return all coordinates
        return list(itertools.product(range(self.x, self.x + self.w + 1), range(self.y, self.y + self.h+1)))


class Pedestrain:
    def __init__(self, coord):
        if coord[0] == 0 or coord[1] == 0 or coord[0]==dim_x-1 or coord[1]==dim_y-1:
            raise Exception("Wall cells included")
        self.status = 0  # 1 if pedestrain exited successfuly, 2 if pedestrain died in the fire
        self.now = coord
        self.update_xy()
        if self.x == 0 or self.y == 0:
            raise Exception("Wall cells included")
        if visual_field[coord] == 999:
            raise Exception("There is already a pedestrain at {}".format(self.now))
        self.last = coord  # last position
        visual_field[coord] = 999

    def step(self):
        print("\nPedstrain: ", self.now)
        if self.now in exit_cells: # exit successfully
            self.status = 1
            visual_field[self.now] = 0
        elif self.now in fire_cells: # died
            self.status = 2
            visual_field[self.now] = 1000
        else:
            visual_field[self.now] = 0
            if self.last != self.now:
                dff_diff[self.last] += 1
            max = np.max(self.P)
            max_index = np.where(self.P == max)
            dir = (max_index[0][0] - 1, max_index[1][0] - 1)
            self.last = self.now
            temp = tuple(self.now + np.array(dir))
            while temp in occupied_cells or temp in fire_cells:
                self.P[self.P == max] = 0
                max = np.max(self.P)
                max_index = np.where(self.P == max)
                dir = (max_index[0][0] - 1, max_index[1][0] - 1)
                temp = tuple(self.now + np.array(dir))
                # print("\nPedestrain: ", self.last, self.now, "\n S:\n", self.get_S(), "\n I:\n", self.I, "\n n:\n",
                #       self.n, "\n epsilon:\n", self.epsilon,
                #       "\n F:\n", self.F, "\n", self.P)
            self.now = temp
            self.update_xy()
            occupied_cells.append(self.now)
            visual_field[self.now] = 999
            print(self.now)

    def update(self):
        if self.now not in exit_cells and self.now not in fire_cells:
            self.update_I()
            self.update_F()
            self.update_epsilon()
            self.update_n()
            self.update_H()
            self.update_Pc()
            self.update_P()

    def update_P(self):  # overall probability
        print( "\nPedestrain: ", self.last, self.now, "\n S:\n", self.get_S(), "\n I:\n", self.I, "\n n:\n", self.n, "\n epsilon:\n", self.epsilon,
              "\n F:\n", self.F, "\n D:\n", self.get_D(),"\n")
        self.P = (np.exp(ks * self.get_S()) * np.exp(kd * self.get_D()) * self.I * (1 - self.n) * self.epsilon) / np.exp(
            kf * self.F)
        print("P: \n", self.P)

    @staticmethod
    def update_cell():
        for i in pedestrains:
            visual_field[i.now] = 999

    def __str__(self):
        return str(self.now)

    def update_xy(self):
        self.x = self.now[0]
        self.y = self.now[1]

    def get_I(self):
        print("I:", self.I)
        return self.I

    def update_I(self):  # I -> inertia filed
        self.I = np.ones((3, 3), dtype=np.longdouble)
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
        neighbors = get_neighbors_including_wall(self.now, 1)
        x = self.x - 1
        y = self.y - 1
        neighbors.append(self.now)
        for i in neighbors:
            self.F[i[0] - x, i[1] - y] = self.compute_H(i)
        # print("F____________", self.F)
        self.F = np.where(self.F == 0, 999999,  1 / self.F) # Fire field overlap with fire cells, to avoid 1/0 error ,set it to 0.01
        # print("F____________", self.F)
        sum = 0
        for i in self.F.flatten():
            sum += i
        self.F = self.F / sum

    def get_F(self):
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
        return self.H

    def get_closet_fire(self):
        return self.closet_fire_cell

    def update_Pc(self):  # Ｐｃ ＝１－ｅｘｐ（－λ（Ｈ －１））
        self.Pc = 1 - math.exp(-myLambda * (self.H - 1))

    def get_Pc(self):
        return self.Pc

    def update_epsilon(self):  # update obstacle matrix,  epsilon is [] when pedestrian on the border/exit
        temp = visual_field[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        self.epsilon = np.where(temp == 500, 0, 1)
        if self.epsilon.size == 0:  # epsilon is [] when pedestrian on the border/exit
            self.epsilon = numpy.zeros((3, 3))

    def get_epsilon(self):
        return self.epsilon

    def update_n(self):  # update target cell occupation matrix, n is [] when pedestrian on the border/exit
        temp = visual_field[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        # print(temp)
        self.n = np.where(temp == 999, 1, 0)
        if self.n.size != 0:  # n is [] when pedestrian on the border/exit
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


def generate_pedestrain_rand(num, rectangle):
    pedestrain_cells = random.sample(rectangle.all_coordinates(), num)
    for i in pedestrain_cells:
        if visual_field[i] == 0:
            pedestrains.append(Pedestrain(i))


def generate_pedestrain(x):  # x is a tuple|Rectangle
    if isinstance(x, Rectangle):
        for i in x.all_coordinates():
            if visual_field[i] == 0:
                pedestrains.append(Pedestrain(i))
    if isinstance(x, tuple):
        if x in fire_cells:
            raise Exception("Fire cells included")
        if x[0] == 0 or x[1] == 0 or x[0]==dim_x-1 or x[1]==dim_y-1:
            raise Exception("Wall cells included")
        pedestrains.append(Pedestrain(x))


import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

def Update(frame, img,):
    if frame == 0: # skip 0
        return img
    one_step(frame)
    img.set_data(visual_field)
    return img


def animate():
    light_gray = np.array([220 / 256, 220 / 256, 220 / 256, 1])
    green = np.array([50 / 256, 200 / 256, 50 / 256, 1])
    red = np.array([255 / 256, 90 / 256, 90 / 256, 1])
    coffee = np.array([111 / 256, 78 / 256, 55 / 256, 1])
    black = np.array([0, 0, 0, 1])
    newColors = np.zeros((5, 4))
    newColors[0, :] = light_gray
    newColors[1, :] = red
    newColors[2, :] = coffee
    newColors[3, :] = green
    newColors[4, :] = black

    Cmap = ListedColormap(newColors)
    boundary_norm = BoundaryNorm([-0.5, 498.5, 499.5, 500.5, 999.5, 1000.5], Cmap.N)

    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(visual_field, cmap=Cmap, interpolation='nearest', norm=boundary_norm)

    ani = animation.FuncAnimation(fig, Update, fargs=(img, ), init_func=init,
                                  frames=10,
                                  interval=300,
                                  repeat=False)
    f = r'.\test.mp4'
    ani.save(f, writer='ffmpeg', fps=1)


def init():
    global fire_cells, exit_cells
    # define exits
    exit_cells = frozenset((
        (dim_x // 2 - 1, dim_y - 1), (dim_x // 2, dim_y - 1),
        # (dim_x - 1, dim_y // 2), (dim_x - 1, dim_y // 2 - 1),
        # (0, dim_y // 2 - 1), (0, dim_y // 2),
        # (dim_x // 2 - 1, 0), (dim_x // 2, 0),
    ))
    # fire_cells = {(4, 4), (4, 5), (5, 4), (5, 5)}
    rec_fire = Rectangle(int((dim_x-2)/2-1), int((dim_y-2)/2-1),1,1)
    fire_cells = set(rec_fire.all_coordinates())
    init_walls(exit_cells)
    init_sff(exit_cells)
    #Assign pedestrains
    rec = Rectangle(1, 1, dim_x-3, dim_y-3)
    generate_pedestrain_rand(40,rec)


def one_step(time):
    global pedestrains, visual_field, occupied_cells
    occupied_cells = []
    print("\n-----------------------------", time)
    fire_evolution(time)
    temp = np.copy(visual_field)
    for i in pedestrains:
        if i.status == 0:
            i.update()
            # i.step()
            # print(visual_field)
    for i in pedestrains:
        if i.status == 0:
            i.step()
            print(temp)
            print(visual_field)
    update_dff()
    init_dff_diff()


def test():
    init()
    for i in range(3):
        one_step(i)

animate()
# test()