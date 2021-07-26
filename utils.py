"""
General Utilities
"""
import random

def gen_rand_constrained(constrain_list, obj_dim, max_dim):
    """
    Generates random numbers which are not in constrain_list
    """
    rand_num = 0
    # Keep looking for random numbers which are not in constrain_list
    while rand_num in constrain_list:
        rand_num = random.randint(1, max_dim - obj_dim)
    return rand_num


def gen_rand_coord(coord_list, obj_dim, max_dim):
    coord = gen_rand_constrained(coord_list, obj_dim, max_dim)
    coord_list.extend([x for x in range(coord, coord + (2 * obj_dim))])
    return coord, coord_list