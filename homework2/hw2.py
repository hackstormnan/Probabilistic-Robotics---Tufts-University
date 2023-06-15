import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import quantitative as q



def particle_filter(num_particles):
        
    # Load the map image and get its dimensions
    map_image = cv2.imread("BayMap.png")
    map_x_range = map_image.shape[1]
    map_y_range = map_image.shape[0]
    # map_x_range, map_y_range, channels = map_image.shape

    # Initialize the drone position and movement uncertainty
    drone_x = 0
    drone_y = 0
    sigma_movement = 10


    drone_x = np.random.randint(0, map_x_range)
    drone_y = np.random.randint(0, map_y_range)

    # generate a bunch of random dots
    # num_particles = 500
    particles = np.zeros((num_particles, 2))
    weights = np.zeros(num_particles)

    cropped_size = 75
    smallest_norm_list = []
    x_labels = []
    smallest_norm = 0


    # Start the simulation
    for i in range(10):
        # Generate a random movement vector [dx, dy] such that dx^2 + dy^2 = 1.0
        # Generate two random numbers between -1 and 1
        movement_x, movement_y = np.random.uniform(low=-1.0, high=1.0, size=2)

        # Normalize the resulting vector
        magnitude = np.sqrt(movement_x**2 + movement_y**2)
        movement_x /= magnitude
        movement_y /= magnitude


        # Check if the movement vector moves the drone off the map
        if abs(drone_x + movement_x) > map_x_range or abs(drone_y + movement_y) > map_y_range:
            # Reject the vector and generate a new random vector
            continue
        

        # Update the drone position with noise
        drone_x += movement_x + np.random.normal(0, sigma_movement)
        drone_y += movement_y + np.random.normal(0, sigma_movement)


        # Draw a circle on the map image at the true position of the drone
        observation_image = map_image.copy()
        cv2.circle(observation_image, (int(drone_x), int(drone_y)), 40, (0, 0, 255), -1)

        # Display the observation and reference images, and wait for user input
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1.imshow(observation_image)
        # ax1.set_title('Observation')
        # ax2.imshow(map_image)
        # ax2.set_title('Reference')


        # Define the rectangular ROI for the crop
        drone_x, drone_y, w, h = int(drone_x), int(drone_y), cropped_size, cropped_size


        # Crop the image
        cropped_map_image = map_image[drone_y:drone_y+h, drone_x:drone_x+w]
        # Check if the movement vector moves the drone off the map
        if abs(drone_x + w) > map_x_range or abs(drone_y + h) > map_y_range:
            # Reject the vector and generate a new random vector
            continue


        # Plot the cropped reference image
        # fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        # ax1.imshow(cropped_map_image)
        # ax1.set_title('Observation_croped')
        # plt.show()


        particle_observation_image = cv2.imread("BayMap.png")

        # add some noise to the cropped reference image (extra credit 3)
        # subsititure part starts

        # noise = np.zeros(cropped_map_image.shape, np.uint8)
        # cv2.randn(noise, 0, 20)  # Adjust the standard deviation (20 in this example) as desired

        # # Add the noise to the image
        # noised_cropped_map_image = cv2.add(cropped_map_image, noise)

        # # # Display the image with and without noise
        # # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # # ax1.imshow(noised_cropped_map_image)
        # # ax1.set_title('noised_cropped_map_image')
        # # ax2.imshow(cropped_map_image)
        # # ax2.set_title('cropped_map_image')


        # cropped_map_image = noised_cropped_map_image

        # subsititure part ends
        

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1.imshow(cropped_map_image)
        # ax1.set_title('Observation_croped (Drone camera)')
        # # ax2.imshow(noised_cropped_map_image)
        # # ax2.set_title('Noised_observation_croped')
        # plt.show()

            
        # generate new particles
        for j in range(num_particles):
            if i == 0:
                particles[j, 0] = np.random.randint(0, map_x_range)
                particles[j, 1] = np.random.randint(0, map_y_range)
            
            particle_x, particle_y, w, h = int(particles[j, 0]), int(particles[j, 1]), cropped_size, cropped_size
            particle_x = int(particle_x)
            particle_y = int(particle_y)
            
            # # Check if the movement vector moves the drone off the map
            # if abs(particle_x+w) > map_x_range:
            #     particle_x -= cropped_size
            # if abs(particle_y+h) > map_y_range:
            #     particle_y -= cropped_size
                


            cropped_particle_image = map_image[particle_y:particle_y+h, particle_x:particle_x+w]
            # fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
            # ax1.imshow(cropped_particle_image)

            # get the weight for different particles (normal)
            # subsititure part starts
            # print(cropped_map_image.shape)
            # print(cropped_particle_image.shape)
            # Ensure that the two images have the same size
            if cropped_particle_image.shape != cropped_map_image.shape:
                cropped_particle_image = cv2.resize(cropped_particle_image, (cropped_map_image.shape[1], cropped_map_image.shape[0]))

            mse = np.mean((cropped_particle_image - cropped_map_image)**2)
            weight = 1 / (mse + 1e-10)
            weight *= 100

            # subsititure part ends

            # # get the weight for different particles (using color histogram, extra credit 2)
            # subsititure part starts

            # particle_hsv = cv2.cvtColor(cropped_particle_image, cv2.COLOR_BGR2HSV)
            # ref_hsv = cv2.cvtColor(cropped_map_image, cv2.COLOR_BGR2HSV)
            
            # # Define the histogram parameters
            # histSize = [16, 16]   # number of bins for each channel
            # h_ranges = [0, 180]   # hue range
            # s_ranges = [0, 256]   # saturation range
            # ranges = h_ranges + s_ranges

            # # Calculate the histograms
            # particle_hist = cv2.calcHist([particle_hsv], [0, 1], None, histSize, ranges)
            # ref_hist = cv2.calcHist([ref_hsv], [0, 1], None, histSize, ranges)

            # # Normalize the histograms
            # particle_hist_norm = cv2.normalize(particle_hist, particle_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # ref_hist_norm = cv2.normalize(ref_hist, ref_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # # Calculate the Bhattacharyya distance
            # bhattacharyya = cv2.compareHist(particle_hist_norm, ref_hist_norm, cv2.HISTCMP_BHATTACHARYYA)

            # # Calculate the weight for each particle
            # weight = 1 / (bhattacharyya + 1e-10)

            # subsititure part ends
            # extra credit 2 ends

            
            weights[j] = weight

            # draw the different size circle for different weight
            radius = int(weight * 10)
            color = (0, 0, 255)
            cv2.circle(particle_observation_image, (particle_x, particle_y), radius, color, -1)

        smallest_norm = q.get_norm(particles, drone_x, drone_y)
        smallest_norm_list.append(smallest_norm)
        x_labels.append(i)
            
        # normalize the weights
        weights_norm = weights / np.sum(weights)


        # compute the cumulative sum of the normalized weights
        cumsum_norm = np.cumsum(weights_norm)


        # generate random numbers for each particle
        rand_nums = np.random.uniform(size=num_particles)


        # initialize the new set of particles
        new_particles = np.zeros_like(particles)

        # iterate over the particles and select a new set of particles based on the random numbers
        j = 0
        for k in range(num_particles):
            while cumsum_norm[j] < rand_nums[k]:
                j += 1

            new_particles[k] = particles[j]

        # update the set of particles with the new set
        particles = new_particles


        # add movement to the particles (normal)
        # subsititure part starts

        for i in particles:
            i[0] += movement_x
            i[1] += movement_y

        # subsititure part ends



        # # add movement to the particles (add noise, extra credit 1)
        # subsititure part starts

        # for i in particles:
        #     i[0] += movement_x + np.random.normal(0, 10)
        #     i[1] += movement_y + np.random.normal(0, 10)

        # subsititure part ends



        # fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        # ax1.imshow(particle_observation_image)


        # plt.tight_layout()
        # plt.show(block=False)
        # input("Press Enter to continue...")
        # plt.close()  # Close the plot window to avoid memory 
    return smallest_norm

    # print(smallest_norm)
    # create the plot Smallest Particle vs Drone Position
    # plt.figure()
    # plt.plot(x_labels, smallest_norm_list)
    # plt.xlabel('Time')
    # plt.ylabel('Norm between Smallest Particle and Drone Position')

    # # set the title
    # plt.title('Smallest Particle vs Drone Position')
    # plt.show()


    # return smallest_norm


def main():
    norm_list = []
    x_labels = []
    i = 100
    while i <= 1000:
        norm_list.append(particle_filter(i))
        x_labels.append(i)
        i += 100
    

    plt.figure()
    plt.plot(x_labels, norm_list)
    plt.xlabel('Particle numbers')
    plt.ylabel('Norm between Smallest Particle and Drone Position')

    # set the title
    plt.title('Smallest Particle vs Drone Position')
    plt.show()

main()















    