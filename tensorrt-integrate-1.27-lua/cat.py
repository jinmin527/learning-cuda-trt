from enum import Enum
import numpy as np

class CatType(Enum):
    Normal = 0
    Escape = 1

class Cat:
    def __init__(self, name, x, y, ori_x, ori_y, ctype, speed=2):
        self.name = name
        self.x = x
        self.y = y
        self.ori_x = ori_x
        self.ori_y = ori_y
        self.speed = speed
        self.ctype = ctype

def get_overlap_cat_pair(cats):

    pairs = []
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            icat = cats[i]
            jcat = cats[j]

            if icat.ctype != jcat.ctype:
                distance = np.sqrt((icat.x - jcat.x) ** 2 + (icat.y - jcat.y) ** 2)
                if distance < 1:
                    pairs.append([icat, jcat])
    return pairs


cmap = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])

cats = [
    Cat("A", 3, 1, 1, 0, CatType.Normal),
    Cat("B", 3, 6, 0, -1, CatType.Escape),
    Cat("C", 7, 8, -1, 0, CatType.Normal)
]

epochs    = 100
max_speed = 4
fps       = max_speed

for t in range(epochs):

    for i in range(fps):

        # check
        pairs = get_overlap_cat_pair(cats)
        if len(pairs) > 0:
            for a, b in pairs:
                tmp_type = a.ctype
                a.ctype = b.ctype
                b.ctype = tmp_type

                if a.ctype == CatType.Normal:
                    a.ori_x *= -1
                    a.ori_y *= -1
                elif b.ctype == CatType.Normal:
                    b.ori_x *= -1
                    b.ori_y *= -1

        # update