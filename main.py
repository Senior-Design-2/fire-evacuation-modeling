import itertools
import math
import numpy
import numpy as np
from math import sqrt

frame = 500  # total time steps
file_name = r'.\test4.mp4'
dim_x = 18
dim_y = 65

tf = 20  # fire evolution frequency
w = 1.05  # w is a weight for I in the paper
ks = 0.5
kd = 10
kf = 0.3
alpha = 0.2  # coeffision for diffusion
delta = 0.2  # coeefision for decay
myLambda = 0.5  # 0.5 # Sedation probability increase coefficient, ps: lambda increase sedation probability increase
gamma = 30  # If distance to fire border is larger than γ , Fij = 0,

pedestrains = []
visual_field = np.zeros((dim_x, dim_y))
sff = np.zeros((dim_x, dim_y))
dff = np.zeros((dim_x, dim_y))  # dynamic floor field


# initialize walls to be 99999
def init_walls(exit_cells):
    sff[0, :] = sff[-1, :] = sff[:, -1] = sff[:, 0] = 99999
    visual_field[0, :] = visual_field[-1, :] = visual_field[:, -1] = visual_field[:, 0] = 99999
    # initialize exit
    for e in exit_cells:
        sff[e] = 0
        visual_field[e] = 0


def init_obstal(obstal):
    for i in obstal:
        sff[i] = 99999
        visual_field[i] = 99999


# get diagonal neighbors of a cell, return a list of cells
def get_diag_neighbors(cell):
    neighbors = []
    i, j = cell
    if i >= 1 and j >= 1 and sff[(i - 1, j - 1)] != 99999 and sff[(i + 1, j)] != 499:
        neighbors.append((i - 1, j - 1))
    if i < dim_x - 1 and j < dim_y - 1 and sff[(i + 1, j + 1)] != 99999 and sff[(i + 1, j)] != 499:
        neighbors.append((i + 1, j + 1))
    if i < dim_x - 1 and j >= 1 and sff[(i + 1, j - 1)] != 99999 and sff[(i + 1, j)] != 499:
        neighbors.append((i + 1, j - 1))
    if i >= 1 and j < dim_y - 1 and sff[(i - 1, j + 1)] != 99999 and sff[(i + 1, j)] != 499:
        neighbors.append((i - 1, j + 1))
    return neighbors


# get neighbors of a cell, default to be von Neumann neighborhood, if second argument = 1, then get moore neighbor
# return a list of cells
def get_neighbors(cell, moore=0, ignore_val=99999):
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
    global sff, dim_x, dim_y
    for e in exit_cells:
        e_neighbor = get_neighbors(e)
        for c in e_neighbor:
            if c not in exit_cells:
                init_sff_rec(c, 1)
    print(sff)
    # sff = np.where(sff==99999,0,1/sff)


# a recursive function to initialize static floor field
def init_sff_rec(_cell, _value):
    global sff
    sff[_cell] = _value
    neighbors = get_neighbors(_cell, moore=0)
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
    global dff

    tmp = np.zeros((dim_x, dim_y))  # tmp matrix to store the result of alpha*(1-delta)/4*sum(dff[neighbors])
    for idx, x in np.ndenumerate(tmp):
        neighbors = get_neighbors(idx)
        for cell in neighbors:
            tmp[idx] += dff[cell]

        tmp[idx] = tmp[idx] * alpha * (1 - delta) / 4

    dff = (1 - alpha) * (1 - delta) * dff + tmp
    dff = dff / np.sum(dff)  # normalize dff



# update fire
def update_fire():
    global sff
    '''todo: further add more rules'''
    for i in fire_cells:
        sff[i] = 99999
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
            neighbors = get_neighbors(i, 1, 99999)
            for j in neighbors:
                tmp.add(j)
        fire_cells = fire_cells.union(tmp)
        update_fire()
        sff[sff != 99999] = 0
        init_sff(exit_cells)


# fire_evolution(20)
# print(fire_cells)
# fire_evolution(30)
# print(fire_cells)

class Rectangle:  # [A,B]
    # [C,D]
    def __init__(self, X, Y, H, W):  # (x,y) are left most coordinates, H height, W width of the box

        self.x = X

        self.y = Y

        self.h = H

        self.w = W

        # check range
        co = self.all_coordinates()
        for i in co:
            if i[0] == 0 or i[1] == 0 or i[0] == dim_x - 1 or i[1] == dim_y - 1:
                raise Exception("Wall cells {} included".format(i))

    def complete(self):
        return self.x, self.y, self.h, self.w

    def range(self):  # return {A,B,C,D}
        return [(self.x, self.y), (self.x + self.h, self.y), (self.x, self.y - self.w),
                (self.x + self.h, self.y - self.w)]

    def all_coordinates(self):  # return all coordinates
        return list(itertools.product(range(self.x, self.x + self.h + 1), range(self.y, self.y + self.w + 1)))


class Pedestrain:
    def __init__(self, coord):
        if coord[0] == 0 or coord[1] == 0 or coord[0] == dim_x - 1 or coord[1] == dim_y - 1:
            raise Exception("Wall cells included")
        self.status = 0  # 1 if pedestrain exited successfuly, 2 if pedestrain died in the fire
        self.now = coord
        self.update_xy()
        if visual_field[coord] == 999:
            raise Exception("There is already a pedestrain at {}".format(self.now))
        self.last = coord  # last position
        visual_field[coord] = 999

    def step(self, decision="max"):
        assert decision == "max" or decision == "probability"
        print("\nPedstrain: ", self.last, self.now)
        if self.now in exit_cells:  # exit successfully
            self.status = 1
            visual_field[self.now] = 0
        elif self.now in fire_cells:  # died
            self.status = 2
            visual_field[self.now] = 1000
        else:
            visual_field[self.now] = 0
            self.last = self.now

            if random.random() > self.Pc and not self.in_catwalk():  # Pedestrian is panic
                print("Pc:", self.Pc, self.F)
                neighors = get_neighbors_including_wall(self.now, 1)
                dic = {}
                for i in neighors:
                    projection_norm = np.dot(np.array(i) - self.now,
                                             np.array(self.closet_fire_cell) - self.now) / np.sqrt(
                        sum((np.array(i) - self.now) ** 2))
                    dic[tuple(np.array(i) - self.now)] = projection_norm
                print(dic)
                dir = min(dic, key=dic.get)  # return the key of the minimum value
                temp = tuple(self.now + np.array(dir))
                while visual_field[temp] != 0 or temp in occupied_cells:  # find aviliable cell
                    dic.pop(dir)
                    if not dic:  # dic is empty, stay
                        occupied_cells.append(self.now)
                        visual_field[self.now] = 998
                        print(self.now)
                        return 0
                    dir = min(dic, key=dic.get)
                    temp = tuple(self.now + np.array(dir))
                self.now = temp
                self.update_xy()
                occupied_cells.append(self.now)
                visual_field[self.now] = 998
                print(self.now)

            else:
                print("\n S:\n", self.get_S(), "\n I:\n", self.I, "\n n:\n",
                      self.n,
                      "\n epsilon:\n", self.epsilon,
                      "\n F:\n", self.F, "\n D:\n", dff, "\n P: \n", self.P)
                if decision == "probability":
                    index = np.random.choice(9, p=self.P.flatten())
                    indexes = [i for i, x in np.ndenumerate(self.P)]
                    next = indexes[index]
                    dir = (next[0] - 1, next[1] - 1)
                    temp = tuple(self.now + np.array(dir))
                    while temp in occupied_cells:
                        index = np.random.choice(9, p=self.P.flatten())
                        next = indexes[index]
                        dir = (next[0] - 1, next[1] - 1)
                        temp = tuple(self.now + np.array(dir))
                    self.now = temp
                    self.update_xy()
                    occupied_cells.append(self.now)
                    visual_field[self.now] = 999
                    print(self.now)
                else:
                    max = np.max(self.P)
                    index = np.where(self.P == max)
                    max_index = (index[0][0], index[1][0])
                    dir = (max_index[0] - 1, max_index[1] - 1)
                    temp = tuple(self.now + np.array(dir))
                    while temp in occupied_cells or temp in fire_cells:
                        self.P[max_index] = 0
                        max = np.max(self.P)
                        index = np.where(self.P == max)
                        max_index = (index[0][0], index[1][0])
                        dir = (max_index[0] - 1, max_index[1] - 1)
                        temp = tuple(self.now + np.array(dir))
                        # print("\nPedestrain: ", self.last, self.now, "\n S:\n", self.get_S(), "\n I:\n", self.I, "\n n:\n",
                        #       self.n,
                        #       "\n epsilon:\n", self.epsilon,
                        #       "\n F:\n", self.F, "\n D:\n", self.get_D(), "\n", self.P)
                        # print(temp)
                    self.now = temp
                    self.update_xy()
                    occupied_cells.append(self.now)
                    visual_field[self.now] = 999
                    print(self.now)

        if self.last != self.now:
            dff[self.last] += 1

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
        self.P = (np.exp(ks * self.get_S()) * np.exp(kd * dff[self.now]) * self.I * (
                1 - self.n) * self.epsilon) / np.exp(
            kf * self.F)
        sum = np.sum(self.P)
        self.P = self.P / sum

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
        print("F: ", self.F)
        s = sff[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        cof1 = np.where(self.F > gamma, 0, 1)  # rule 1 coffession
        print("cof1", cof1)

        cof2 = np.where(s < 6, 0.5, 1)  # rule 2 coffession
        print("cof2", cof2)

        mask = np.nonzero(self.F)
        self.F[mask] = 1 / self.F[mask]
        self.F[self.F == 0] = 1000  # Fire field overlap with fire cells, to avoid 1/0 error ,set it to 0.01
        self.F = self.F * cof1
        sum = 0
        for i in self.F.flatten():
            if i != 1000:
                sum += i
        if sum != 0:
            self.F = np.where(self.F != 1000, self.F / sum, 1000)
        self.F = self.F * cof2

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
        self.epsilon = np.where(temp == 99999, 0, 1)
        temp = np.where(temp == 499, 0, 1)
        self.epsilon[temp == 0] = 0
        if self.epsilon.size == 0:  # epsilon is [] when pedestrian on the border/exit
            self.epsilon = numpy.zeros((3, 3))

    def get_epsilon(self):
        return self.epsilon

    def update_n(self):  # update target cell occupation matrix, n is [] when pedestrian on the border/exit
        temp = visual_field[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        temp[temp == 998] = 999
        self.n = np.where((temp == 999), 1, 0)
        self.n[1, 1] = 0
        # if self.n.size != 0:  # n is [] when pedestrian on the border/exit
        #     self.n[1, 1] = 0
        # else:
        #     self.n = numpy.ones((3, 3))

    def get_n(self):
        return self.n

    def get_S(self):
        s = sff[self.x - 1:self.x + 2, self.y - 1:self.y + 2]
        print(s)
        s = s[1, 1] - s
        for i in range(3):
            for j in range(3):
                if i + j == 0 or i + j == 2 or i + j == 4:
                    s[i][j] = s[i][j] / sqrt(2)
        return s


    def in_catwalk(self):  # decide if pedestrian is in a catwalk, return T, F
        obstacle = self.epsilon
        count = (obstacle == 1).sum()
        print(obstacle)
        if count <= 4:
            return True
        return False


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
        if x[0] == 0 or x[1] == 0 or x[0] == dim_x - 1 or x[1] == dim_y - 1:
            raise Exception("Wall cells included")
        pedestrains.append(Pedestrain(x))


import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

# Statistic variable
time = []
left = []
dead = []
exited = []
panic = []


def record(frame):
    global left, dead, time, exited, panic
    count = 0
    count_dead = 0
    count_exited = 0
    for i in pedestrains:
        if i.status == 0:
            count += 1
        elif i.status == 1:
            count_exited += 1
        else:
            count_dead += 1

    count_panic = (visual_field == 998).sum()
    time.append(frame)
    left.append(count)
    dead.append(count_dead)
    exited.append(count_exited)
    panic.append(count_panic)


def plot():
    fig, ax = plt.subplots()
    ax.set_xlabel('time step')
    ax.set_ylabel('num of people')
    ax.plot(time, left,
            color='b',
            linewidth=1.0,
            label="people left"
            )
    ax.plot(time, dead,
            color='r',
            linewidth=1.0,
            label="people dead"
            )
    ax.plot(time, exited,
            color='g',
            linewidth=1.0,
            label="people exited"
            )
    ax.plot(time, panic,
            color='m',
            linewidth=1.0,
            label="people panic"
            )
    ax.legend()
    plt.savefig(file_name[:-3] + "png")


# Animation
def Update(frame, img, ax):
    record(frame)
    if frame == 0:  # skip 0
        ax.set_title(frame)
        img.set_data(visual_field)
        return img
    one_step(frame)
    ax.set_title(frame)
    img.set_data(visual_field)

    return img


def animate():
    Cmap = ListedColormap(['w', 'r', 'm', 'g', 'k', 'peru'])
    boundary_norm = BoundaryNorm([-0.5, 498.5, 499.5, 998.5, 999.5, 1000.5, 99999.5], Cmap.N)

    fig, ax = plt.subplots()
    ax.axis('off')
    img = ax.imshow(visual_field, cmap=Cmap, interpolation='nearest', norm=boundary_norm)

    ani = animation.FuncAnimation(fig, Update, fargs=(img, ax,), init_func=init,
                                  frames=frame,
                                  interval=200,
                                  repeat=False)
    f = file_name
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
    rec_fire = Rectangle(int((dim_x - 2) / 2), int((dim_y - 2) / 2) - 5, 1, 1)
    fire_cells = set(rec_fire.all_coordinates())
    update_fire()
    init_walls(exit_cells)

    # Assign obstacle
    obstacal = Rectangle(10, int(dim_y / 2), 6, 1)
    init_obstal(obstacal.all_coordinates())

    # sff
    init_sff(exit_cells)
    # Assign pedestrains
    rec = Rectangle(1, 1, dim_x - 3, dim_y - 3)
    generate_pedestrain_rand(200, rec)
    # generate_pedestrain(((5, 1)))
    print(sff)
    print(dff)


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
            i.step("max")
            print(temp)
            print(visual_field)

    update_dff()
    print("D: \n", dff)


def test():
    init()
    for i in range(4):
        one_step(i)


animate()
plot()
# test()
# exit_cells = frozenset((
#         (dim_x // 2 - 1, dim_y - 1), (dim_x // 2, dim_y - 1),
#         # (dim_x - 1, dim_y // 2), (dim_x - 1, dim_y // 2 - 1),
#         # (0, dim_y // 2 - 1), (0, dim_y // 2),
#         # (dim_x // 2 - 1, 0), (dim_x // 2, 0),
#     ))
# init_walls(exit_cells)
# init_sff(exit_cells)
# print(1)
