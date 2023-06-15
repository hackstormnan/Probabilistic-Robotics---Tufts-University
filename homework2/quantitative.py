import numpy as np

# get the smallest norm
def get_norm(particles, drone_x, drone_y):
    drone_position = np.array([drone_x, drone_y])
    closest_norm = 9999999
    for particle in particles:
        norm = np.linalg.norm(np.array(particle) - drone_position)
        closest_norm = min(closest_norm, norm)
    
    return closest_norm









