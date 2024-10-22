import math

def calculate_distance(pos1, pos2):
    """
    Calculate the 2D distance between two positions.
    pos1, pos2: Lists or tuples containing x and y coordinates.
    """
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
