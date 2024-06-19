import numpy as np
import cv2
from pathlib import Path
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from skimage.measure import shannon_entropy
from skimage import filters
from brisque import BRISQUE
import os

# Function to load and prepare the image
def load_and_prepare_image(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("The image file was not found.")
    return image

# Initialize random seed based on current time
def initialize_random_seed():
    current_time = time.localtime(time.time())
    random_seed = sum([i * 25000000 for i in current_time[3:6]])
    random.seed(random_seed)
    np.random.seed(random_seed)

# Normalize image intensity
def normalize_image_intensity(image):
    min_input_intensity = np.min(image)
    max_input_intensity = np.max(image)
    print(f"The minimum pixel intensity of the input image is {min_input_intensity}, and the maximum is {max_input_intensity}")
    print(f"Average pixel intensity: {np.mean(image)}")
    min_input_intensity_plus_1 = min_input_intensity + 1
    max_input_intensity_minus_1 = max_input_intensity - 1
    relative_input_range = (max_input_intensity - min_input_intensity) / 255
    # image = np.clip(image, 0, None)
    return min_input_intensity, max_input_intensity,  min_input_intensity_plus_1 ,max_input_intensity_minus_1 , relative_input_range

# Done because brisque works only on 3 channels images
def make_3_channel_image(image):
    print("\nOriginal shape:", np.shape(image))
    image = np.stack((image,)*3, axis=-1)
    print("\n3 channel shape for brisque:", np.shape(image))
    return image

def make_single_channel_image(image):
    if image.ndim == 3 and image.shape[2] == 3:  # Check if it's a 3-channel image
        print("\n3 channel shape for brisque:", np.shape(image))
        single_channel_image = image[:, :, 0]  # Extract only the first channel
        print("\nOriginal shape:", np.shape(single_channel_image))
        return single_channel_image

# Function to calculate fitness (Placeholder)
def calculate_fitness(input_image, lut_current, fitness_function):
    im_current = cv2.LUT(input_image, lut_current)

    if fitness_function == "classic":
        # Split the image into its three channels (B, G, R)
        channels = cv2.split(input_image)

        # Initialize an empty list to hold the fitness value for each channel
        fitness_values = []

        # Iterate through each channel
        for channel in channels:
            # Apply the lookup table to the current channel
            im_current_channel = cv2.LUT(channel, lut_current)

            # Calculate standard deviation for the current channel
            std_dev = np.std(im_current_channel)

            # Calculate entropy for the current channel
            entropy = shannon_entropy(im_current_channel)
            sobel_h = filters.sobel_h(im_current_channel)
            sobel_v = filters.sobel_v(im_current_channel)
            mean_sobel = np.mean(np.abs(sobel_h) + np.abs(sobel_v))

            # Calculate fitness for the current channel and add it to the list
            fitness_channel = (std_dev * entropy * mean_sobel) ** (1/3)

            # Now, fitness_values contains the fitness score for each of the three channels`
            fitness_values.append(fitness_channel)
        fitness = np.mean(fitness_values)
    

    elif fitness_function == "brisque":
            #Brisque part
            brisque = BRISQUE()
            quality_score = brisque.score(im_current)

            normalized_quality_score = quality_score / 2.0 #make sure the score does not exceed 100

            target_mean = 65

            weight_quality= 0.7
            weight_intensity= 0.3

            # Calculate mean intensity deviation
            current_mean = im_current.mean()
            mean_deviation = abs(current_mean - target_mean)

            # Normalize mean intensity deviation
            normalized_intensity_deviation = mean_deviation / target_mean

            # Combine the scores: Higher fitness is better
            # fitness = (100 - normalized_quality_score) * weight_quality + (100 - normalized_intensity_deviation * 100) * weight_intensity
            fitness =  normalized_quality_score
            if fitness< 0 :
                print(f"\nThe BRISQUE quality score of the image is: {quality_score}, {fitness}")
            
    else:
        print("No fitness function is set or does not met criteria()")
        
    return fitness, im_current

def set_initials(number_iterations):
    """
    Initializes and returns schedules and parameters for SA and GA based on the number of iterations.

    Args:
        number_iterations (int): The total number of iterations for the optimization process.

    Returns:
        dict: A dictionary containing all initialized schedules, parameters, and variables.
    """
    # Schedules for simulated annealing
    SA_schedule = list(range(10, round(40 * number_iterations / 100) + 1, 10)) + \
                  list(range(round(46 * number_iterations / 100), round(70 * number_iterations / 100) + 1, 6)) + \
                  list(range(round(73 * number_iterations / 100), number_iterations + 1, 3))

    # Number of points, ants, and duration for each SA point
    SA_point_num = [2] * int(np.ceil(4 * number_iterations / 100)) + \
                   [4] * int(np.ceil(5 * number_iterations / 100)) + \
                   [6] * int(np.ceil(10 * number_iterations / 100))
    SA_ant_num = [1] * int(np.ceil(4 * number_iterations / 100)) + \
                 [2] * int(np.ceil(5 * number_iterations / 100)) + \
                 [4] * int(np.ceil(10 * number_iterations / 100))
    SA_duration = [3] * int(np.ceil(4 * number_iterations / 100)) + \
                  [6] * int(np.ceil(5 * number_iterations / 100)) + \
                  [12] * int(np.ceil(10 * number_iterations / 100))

    # Schedule for genetic algorithm
    GA_schedule = list(range(5, round(25 * number_iterations / 100) + 1, 5)) + \
                  list(range(round(31 * number_iterations / 100), round(55 * number_iterations / 100) + 1, 6)) + \
                  list(range(round(62 * number_iterations / 100), round(90 * number_iterations / 100) + 1, 7))

    # Initialize variables for the process
    variables = {
        'SA_schedule': SA_schedule,
        'SA_point_num': SA_point_num,
        'SA_ant_num': SA_ant_num,
        'SA_duration': SA_duration,
        'GA_schedule': GA_schedule,
        'SA_interrupt_num': 1,
        'GA_interrupt_num': 1,
        'pheromone_map': np.zeros((256, 256)),
        'best_fitness': 0,
        'gene_fitness': np.zeros(10),
        'genetic_elite': np.zeros(10),
        'iterations_until_GA': 0,
        'best_GA_fitness': 0,
        'enhancement_lut': np.zeros(256),
        'elite_pheromone_trace': np.zeros((256, 256)),
        'fitness_per_iteration': np.zeros(number_iterations),
        'last_enhancing_part': 'Ant Colony Optimization',
        'temperature': 200, # Set initial temperature
        'probability_factors': {
            'up': np.zeros(20),
            'right': np.zeros(20),
            'alpha': np.zeros(20),
            'beta': np.zeros(20),
            'routing': np.zeros(20)
        },
        'best_chromosome': [np.zeros(20) for _ in range(5)]  # Assuming this structure fits your model needs
    }

    # Populate probability factors with random initial values
    indices = np.arange(2, 21, 2)
    factors_scaling = [3, 3, 5, 5, 150]  # Define scaling factors for random values
    for idx, key in enumerate(['up', 'right', 'alpha', 'beta', 'routing']):
        if key == 'routing':
            # Routing factor needs integer values between 1 and 150
            variables['probability_factors'][key][indices - 1] = np.random.randint(1, 151, size=10)
        else:
            # Other factors need scaled floating-point values
            variables['probability_factors'][key][indices - 1] = factors_scaling[idx] * np.random.rand(10)

        # Apply even indices to odd for symmetry
        variables['probability_factors'][key][indices - 2] = variables['probability_factors'][key][indices - 1]

        # Initialize best_chromosome using the initial random values of probability_factors
        variables['best_chromosome'][idx] = variables['probability_factors'][key]

    return variables

def GA(variables):
    """
    Perform a genetic algorithm operation using initialized variables.

    Args:
    - variables (dict): Dictionary containing all necessary settings, parameters, and initialized values.

    Returns:
    - dict: Updated variables with adjusted probability factors and other genetic settings.
    """
    # Extracting probability factors and GA fitness
    probability_factor_vector_up = variables['probability_factors']['up']
    probability_factor_vector_right = variables['probability_factors']['right']
    impact_factor_vector_alpha = variables['probability_factors']['alpha']
    impact_factor_vector_beta = variables['probability_factors']['beta']
    routing_factor_vector = variables['probability_factors']['routing']
    GA_fitness = variables['gene_fitness']

    # Genetic Algorithm logic
    sorted_indices = np.argsort(GA_fitness)
    sorted_GA_fitness = GA_fitness[sorted_indices]

    cum_sum = np.cumsum(sorted_GA_fitness)
    selected_chromosome_1 = sorted_indices[np.searchsorted(cum_sum / cum_sum[-1], np.random.rand())]

    censored_GA_fitness = GA_fitness.copy()
    censored_GA_fitness[selected_chromosome_1] = 0
    censored_sorted_indices = np.argsort(censored_GA_fitness)
    cum_sum_censored = np.cumsum(censored_GA_fitness[censored_sorted_indices])
    selected_chromosome_2 = censored_sorted_indices[np.searchsorted(cum_sum_censored / cum_sum_censored[-1], np.random.rand())]

    parent_index = [selected_chromosome_1, selected_chromosome_2]
    sorted_parent_index = np.argsort(GA_fitness[parent_index])

    worst_chromosome_index = sorted_indices[0]
    if parent_index[sorted_parent_index[0]] == worst_chromosome_index:
        worst_chromosome_index = sorted_indices[1]

    child_1 = [probability_factor_vector_up[parent_index[0]],
               probability_factor_vector_right[parent_index[0]],
               impact_factor_vector_alpha[parent_index[0]],
               impact_factor_vector_beta[parent_index[0]],
               routing_factor_vector[parent_index[0]]]
    child_2 = [probability_factor_vector_up[parent_index[1]],
               probability_factor_vector_right[parent_index[1]],
               impact_factor_vector_alpha[parent_index[1]],
               impact_factor_vector_beta[parent_index[1]],
               routing_factor_vector[parent_index[1]]]

    if np.random.rand() <= 0.85:  # Crossover probability
        for gene_counter in range(5):
            if np.random.rand() < 0.5:  # Gene crossover probability
                child_1[gene_counter], child_2[gene_counter] = child_2[gene_counter], child_1[gene_counter]

    def mutate(child):
        if np.random.rand() < 0.05:  # Mutation probability
            mutation_gene = np.random.randint(5)
            if mutation_gene == 4:  # Routing factor mutation
                mutation_value = np.random.randint(-15, 15)
                child[mutation_gene] = np.clip(child[mutation_gene] + mutation_value, 0, 150)
            elif mutation_gene >= 2:  # Alpha and Beta mutation
                mutation_value = np.random.rand() - 0.5
                child[mutation_gene] = np.clip(child[mutation_gene] + mutation_value, 0, 5)
            else:  # Up and Right probability factors mutation
                mutation_value = (0.4 * np.random.rand()) - 0.2
                child[mutation_gene] = np.clip(child[mutation_gene] + mutation_value, 0, 2)

    mutate(child_1)
    mutate(child_2)

    # Replace a weaker parent with offspring
    replacement_index_1 = 2 * parent_index[sorted_parent_index[0]]
    variables['probability_factors']['up'][replacement_index_1-1:replacement_index_1] = child_1[0]
    variables['probability_factors']['right'][replacement_index_1-1:replacement_index_1] = child_1[1]
    variables['probability_factors']['alpha'][replacement_index_1-1:replacement_index_1] = child_1[2]
    variables['probability_factors']['beta'][replacement_index_1-1:replacement_index_1] = child_1[3]
    variables['probability_factors']['routing'][replacement_index_1-1:replacement_index_1] = child_1[4]

    # Replace the worst chromosome in the population with offspring 2
    replacement_index_2 = 2 * worst_chromosome_index
    variables['probability_factors']['up'][replacement_index_2-1:replacement_index_2] = child_2[0]
    variables['probability_factors']['right'][replacement_index_2-1:replacement_index_2] = child_2[1]
    variables['probability_factors']['alpha'][replacement_index_2-1:replacement_index_2] = child_2[2]
    variables['probability_factors']['beta'][replacement_index_2-1:replacement_index_2] = child_2[3]
    variables['probability_factors']['routing'][replacement_index_2-1:replacement_index_2] = child_2[4]

    return variables



def ACO(variables, min_input_intensity, max_input_intensity, input_image, relative_input_range, lut_current, fitness_function):
    """
    Perform Ant Colony Optimization using initialized variables to find the best Look-Up Table (LUT).

    Args:
    - variables (dict): Dictionary containing all necessary settings, parameters, and initialized values.

    Returns:
    - dict: Updated variables with adjustments based on ACO findings.
    """
    pheromone_map_in = variables['pheromone_map']
    probability_factor_vector_up = variables['probability_factors']['up']
    probability_factor_vector_right = variables['probability_factors']['right']
    impact_factor_vector_alpha = variables['probability_factors']['alpha']
    impact_factor_vector_beta = variables['probability_factors']['beta']
    routing_factor_vector = variables['probability_factors']['routing']

    # Initialize matrices and vectors
    lut_matrix = np.zeros((20, 256), dtype=np.uint8)
    pheromone_trace_matrix = np.zeros((256, 256, 20))
    fitness_vector = np.zeros(20)
    routing_factor_vector_up = relative_input_range * routing_factor_vector

    best_fitness = variables['best_fitness']

    # Iterate over each ant to find the best path
    for ant in range(20):
        intensity = 1
        height = 1
        current_lut = np.zeros(256, dtype=np.uint8)
        pheromone_trace = np.zeros((256, 256))
        pheromone_trace[255, 0] = 1  # Start with an initial pheromone trace

        while not (intensity == 255 and height == 255):
            if intensity < min_input_intensity or intensity > max_input_intensity or height == 255:
                next_point_probability = [0, 0, 1]
            elif intensity == max_input_intensity:
                next_point_probability = [1, 0, 0]
            else:
                a = (pheromone_map_in[255 - height, intensity] + 1) ** impact_factor_vector_alpha[ant]
                b = (probability_factor_vector_up[ant] * (1 + ((intensity - min_input_intensity) / routing_factor_vector_up[ant]) ** 10)) ** impact_factor_vector_beta[ant]
                c = (pheromone_map_in[255 - height, intensity + 1] + 1) ** impact_factor_vector_alpha[ant]
                d = (pheromone_map_in[255 - height, intensity + 1] + 1) ** impact_factor_vector_alpha[ant]
                e = (probability_factor_vector_right[ant] * (1 + (height / routing_factor_vector[ant]) ** 10)) ** impact_factor_vector_beta[ant]
                next_point_probability = [a * b, c, d * e]
                next_point_probability /= np.sum(next_point_probability)
            # Select the next direction based on the calculated probabilities
            sorted_indices = np.argsort(next_point_probability)
            sorted_probability = np.sort(next_point_probability)
            selected_directions = sorted_indices[np.cumsum(sorted_probability) >= np.random.rand()]
            if len(selected_directions) > 0:
                # Move the ant according to the selected direction
                if selected_directions[0] == 0:  # up
                    height += 1
                elif selected_directions[0] == 1:  # upper-right
                    height += 1
                    intensity += 1
                elif selected_directions[0] == 2:  # right
                    intensity += 1
                
            # Update the LUT based on the ant's new position
            current_lut[intensity] = height -1
            # Update the ant's pheromone trace
            pheromone_trace[256-height, intensity-1] = 1

            current_lut[intensity] = height - 1
            pheromone_trace[256-height, intensity-1] = 1

        # Calculate fitness for the path
        fitness, im_current = calculate_fitness(input_image, current_lut, fitness_function)
        fitness_vector[ant] = fitness

        if fitness_vector[ant] > best_fitness:
            variables['best_fitness'] = fitness_vector[ant]
            variables['enhancement_lut'] = current_lut.copy()
            variables['elite_pheromone_trace'] = pheromone_trace.copy()
            variables['last_enhancing_part'] = 'Ant Colony Optimization'

        pheromone_trace_matrix[:, :, ant] = pheromone_trace
        lut_matrix[ant, :] = current_lut

    variables['fitness_vector'] = fitness_vector
    variables['lut_matrix'] = lut_matrix
    variables['pheromone_trace_matrix'] = pheromone_trace_matrix
    
    return variables


def SA(variables, min_input_intensity, max_input_intensity, min_input_intensity_plus_1,  max_input_intensity_minus_1, fitness_function , input_image):
    """
    Perform Simulated Annealing (SA) on image enhancement using initialized variables.

    Args:
    - variables (dict): Contains all the settings and parameters initialized previously.

    Returns:
    - dict: Updated dictionary after performing SA.
    """
    temperature = variables['temperature']
    enhancement_lut = variables['enhancement_lut']
    best_fitness = variables['best_fitness']
    pheromone_trace_matrix = variables['pheromone_trace_matrix']
    elite_pheromone_trace = variables['elite_pheromone_trace']
    fitness_vector = variables['fitness_vector']
    lut_matrix = variables['lut_matrix']
    SA_interrupt_num = variables['SA_interrupt_num']
    SA_point_num = variables['SA_point_num'][SA_interrupt_num - 1]
    SA_ant_num = variables['SA_ant_num'][SA_interrupt_num - 1]
    SA_duration = variables['SA_duration'][SA_interrupt_num - 1]
 
    selected_lut = 20
    lut_matrix = np.vstack([lut_matrix, enhancement_lut])
    fitness_vector = np.append(fitness_vector, best_fitness)
    pheromone_trace_matrix = np.dstack((pheromone_trace_matrix, elite_pheromone_trace))

    lower_bound = 0

    for lut_num in tqdm(range(SA_ant_num + 1)):

        last_point = min_input_intensity - 1

        optimization_lut = lut_matrix[selected_lut, :]
        optimization_fitness = fitness_vector[selected_lut]
        optimization_pheromone_trace = pheromone_trace_matrix[:, :, selected_lut]

        for point_num in range(SA_point_num):

            last_point += np.random.randint(max_input_intensity - SA_point_num + point_num - last_point)
            active_point = last_point

            for SA_cycle in range(SA_duration):

                new_lut = optimization_lut.copy()
                new_pheromone_trace = optimization_pheromone_trace.copy()
                new_active_point = active_point
                break_search = True

                neighborhood_probability = [1, 1, 1, 1, 0, 1, 1, 1, 1]

                if active_point == min_input_intensity:
                    # Restriction of neighborhood at vertical border of LUT
                    neighborhood_probability[0:7:3] = [0, 0, 0]
                    
                    # Restriction of neighborhood to keep LUT monotonically increasing
                    if optimization_lut[active_point] == optimization_lut[active_point + 1]:
                        neighborhood_probability[2] = 0
                    if optimization_lut[active_point] == optimization_lut[active_point + 2]:
                        neighborhood_probability[3] = 0

                elif active_point == max_input_intensity:
                    # Restriction of neighborhood at vertical border of LUT
                    neighborhood_probability[2:9:3] = [0, 0, 0]
                    
                    # Restriction of neighborhood to keep LUT monotonically increasing
                    if optimization_lut[active_point] == optimization_lut[active_point - 1]:
                        neighborhood_probability[8] = 0
                    if optimization_lut[active_point] == optimization_lut[active_point - 2]:
                        neighborhood_probability[7] = 0

                elif active_point == min_input_intensity_plus_1:
                    # Restriction of neighborhood to keep LUT monotonically increasing
                    if optimization_lut[active_point] == optimization_lut[active_point + 1]:
                        neighborhood_probability[1:3] = [0, 0]
                    if optimization_lut[active_point] == optimization_lut[active_point - 1]:
                        neighborhood_probability[8:10] = [0, 0]
                    if optimization_lut[active_point] == optimization_lut[active_point + 2]:
                        neighborhood_probability[3] = 0

                elif active_point == max_input_intensity_minus_1:
                    # Restriction of neighborhood to keep LUT monotonically increasing
                    if optimization_lut[active_point] == optimization_lut[active_point + 1]:
                        neighborhood_probability[1:3] = [0, 0]
                    if optimization_lut[active_point] == optimization_lut[active_point - 1]:
                        neighborhood_probability[8:10] = [0, 0]
                    if optimization_lut[active_point] == optimization_lut[active_point - 2]:
                        neighborhood_probability[7] = 0

                else:
                    # Restriction of neighborhood to keep LUT monotonically increasing
                    if optimization_lut[active_point] == optimization_lut[active_point + 1]:
                        neighborhood_probability[1:3] = [0, 0]
                    if optimization_lut[active_point] == optimization_lut[active_point - 1]:
                        neighborhood_probability[8:10] = [0, 0]
                    if optimization_lut[active_point] == optimization_lut[active_point + 2]:
                        neighborhood_probability[3] = 0
                    if optimization_lut[active_point] == optimization_lut[active_point - 2]:
                        neighborhood_probability[7] = 0

                # Restriction of neighborhood at horizontal border of LUT
                if optimization_lut[active_point] == 0:
                    neighborhood_probability[6:9] = [0, 0, 0]
                elif optimization_lut[active_point] == 255:
                    neighborhood_probability[0:3] = [0, 0, 0]
                
                # Move to new point
                neighbor_num = sum(neighborhood_probability)
                for neighbor_counter in range(neighbor_num):
                    
                    # Randomly suggest a direction to move
                    remaining_neighbor = [i for i, x in enumerate(neighborhood_probability) if x]
                    suggested_direction = remaining_neighbor[np.random.randint(len(remaining_neighbor))]
                    
                    # Remove suggested direction from remaining directions
                    neighborhood_probability[suggested_direction] = 0
                    
                    # Calculate effect of suggested direction
                    if suggested_direction == 1:
                        new_lut[active_point - 1] = optimization_lut[active_point] + 1
                        new_lut[active_point] = optimization_lut[active_point] + 1
                        new_pheromone_trace[:, active_point] = 0  # clean ant's previous way
                        new_pheromone_trace[255 - new_lut[active_point], active_point] = 1  # write ant's new way
                        if active_point != max_input_intensity:
                            new_pheromone_trace[256 - new_lut[active_point], active_point + 1] = 0  # clean ant's previous way
                        new_pheromone_trace[255 - new_lut[active_point]:255 - optimization_lut[active_point - 1], active_point - 1] = 1  # write ant's new way

                    elif suggested_direction == 2:
                        new_lut[active_point] = optimization_lut[active_point] + 1
                        new_pheromone_trace[255 - new_lut[active_point], active_point] = 1  # write ant's new way
                        if active_point != max_input_intensity:
                            new_pheromone_trace[256 - new_lut[active_point], active_point + 1] = 0  # clean ant's previous way

                    elif suggested_direction == 3:
                        if (active_point + 1) < 256 :
                            new_lut[active_point + 1] = optimization_lut[active_point] + 1
                            new_active_point = active_point + 1
                        else:
                            new_lut[active_point] = optimization_lut[active_point] + 1   
                        new_pheromone_trace[255 - optimization_lut[new_active_point]:254 - new_lut[new_active_point], new_active_point] = 0  # clean ant's previous way
                        if new_active_point != max_input_intensity:
                            new_pheromone_trace[255 - optimization_lut[new_active_point + 1]:255 - new_lut[new_active_point], new_active_point + 1] = 1  # write ant's new way
                        new_pheromone_trace[255 - new_lut[new_active_point], new_active_point] = 1  # write ant's new way

                    elif suggested_direction == 4:
                        new_lut[active_point - 1] = optimization_lut[active_point]
                        new_active_point = active_point - 1
                        new_pheromone_trace[255 - new_lut[new_active_point]:255 - optimization_lut[new_active_point], new_active_point] = 1  # write ant's new way
                        new_pheromone_trace[256 - new_lut[new_active_point]:256, active_point] = 0  # clean ant's previous way

                    elif suggested_direction == 6:
                        if (active_point + 1) < 256 :
                            new_lut[active_point + 1] = optimization_lut[active_point]
                            new_active_point = active_point + 1
                        else:
                            new_lut[active_point] = optimization_lut[active_point]
                        new_pheromone_trace[:, new_active_point] = 0  # clean ant's previous way
                        if new_active_point != max_input_intensity:
                            new_pheromone_trace[255 - optimization_lut[new_active_point + 1]:255 - new_lut[new_active_point], new_active_point + 1] = 1  # write ant's new way
                        new_pheromone_trace[255 - new_lut[new_active_point], new_active_point] = 1  # write ant's new way

                    elif suggested_direction == 7:
                        new_lut[active_point - 1] = optimization_lut[active_point] - 1
                        new_active_point = active_point - 1
                        new_pheromone_trace[255 - new_lut[new_active_point]:255 - optimization_lut[new_active_point], new_active_point] = 1  # write ant's new way
                        new_pheromone_trace[256 - new_lut[new_active_point]:256, active_point] = 0  # clean ant's previous way
                        new_pheromone_trace[254 - new_lut[new_active_point], new_active_point] = 0  # clean ant's previous way

                    elif suggested_direction == 8:
                        new_lut[active_point] = optimization_lut[active_point] - 1
                        new_pheromone_trace[255 - new_lut[active_point], active_point] = 1  # write ant's new way
                        if active_point != max_input_intensity:
                            new_pheromone_trace[254 - new_lut[active_point], active_point + 1] = 1  # write ant's new way
                        new_pheromone_trace[254 - new_lut[active_point], active_point] = 0  # clean ant's previous way

                    elif suggested_direction == 9:
                        new_lut[active_point + 1] = optimization_lut[active_point] - 1
                        new_lut[active_point] = optimization_lut[active_point] - 1
                        new_pheromone_trace[255 - new_lut[active_point], active_point] = 1  # write ant's new way
                        new_pheromone_trace[254 - new_lut[active_point], active_point] = 0  # clean ant's previous way
                        new_pheromone_trace[:, active_point + 1] = 0  # clean ant's previous way
                        if active_point != max_input_intensity_minus_1:
                            new_pheromone_trace[255 - optimization_lut[active_point + 2]:255 - new_lut[active_point], active_point + 2] = 1  # write ant's new way
                        new_pheromone_trace[255 - new_lut[active_point], active_point + 1] = 1  # write ant's new way

                    # Decide to select the suggested direction or not
                    new_fitness, new_enhanced = calculate_fitness(input_image, new_lut, fitness_function) # Assuming fitnesscalc is a defined function
                    if new_fitness >= optimization_fitness or np.random.rand() <= np.exp((new_fitness - optimization_fitness) / (0.05 * optimization_fitness) * temperature):
                        optimization_lut = new_lut.copy()
                        optimization_pheromone_trace = new_pheromone_trace.copy()
                        active_point = new_active_point
                        optimization_fitness = new_fitness
                        break_search = False
                        break
                    else:
                        new_active_point = active_point
                        new_lut = optimization_lut.copy()
                        new_pheromone_trace = optimization_pheromone_trace.copy()

                # Check for break search or not
                if break_search:
                    break

                # Search for best fitness
                if optimization_fitness >= best_fitness:
                    
                    best_fitness = optimization_fitness
                    enhancement_lut = optimization_lut.copy()
                    elite_pheromone_trace = optimization_pheromone_trace.copy()
                    variables['last_enhancing_part'] = 'Simulated Annealing'


    variables['enhancement_lut'] = enhancement_lut
    variables['best_fitness'] = best_fitness
    variables['pheromone_trace_matrix'] = pheromone_trace_matrix
    variables['elite_pheromone_trace'] = elite_pheromone_trace
    variables['fitness_vector'] = fitness_vector
    

    return variables

# Usage example:
# variables = set_initials(100)  # Set the initial variables for the optimization
# updated_variables = SA(variables)  # Perform SA with the initialized parameters

def main(args):
    start_time = time.time()
    input_image = load_and_prepare_image(args.image_path)
    # Done because brisque works only on 3 channels images
    input_image = make_3_channel_image(input_image)
    initialize_random_seed()
    min_input_intensity, max_input_intensity, min_input_intensity_plus_1, max_input_intensity_minus_1, relative_input_range = normalize_image_intensity(input_image)
    variables = set_initials(args.number_iterations)

    iterations_until_GA = 0  # Initializing variable to track iterations until GA
    genetic_elite = np.zeros(10)  # Initial elite scores for genetic algorithm
    lut_current = np.zeros(256)  # Initialize LUT for current processing state
    pheromone_map = np.zeros((256, 256))  # Initialize the pheromone map

    for iteration in tqdm(range(1, args.number_iterations + 1)):
        variables = ACO(variables, min_input_intensity, max_input_intensity, input_image, relative_input_range, lut_current, args.fitness_function)

        iterations_until_GA += 1
        pairs = [(i, i + 1) for i in range(0, 20, 2)]
        variables['gene_fitness'] += np.array([sum(variables['fitness_vector'][i:j + 1]) for i, j in pairs])
        genetic_elite = np.maximum(genetic_elite, [max(variables['fitness_vector'][i:j + 1]) for i, j in pairs])

        # Check for SA turn
        if iteration == variables['SA_schedule'][variables['SA_interrupt_num'] - 1] and args.sa_enable:
            print("\nWent into SA")
            variables = SA(variables, min_input_intensity, max_input_intensity, min_input_intensity_plus_1, max_input_intensity_minus_1, args.fitness_function, input_image)
            variables['SA_interrupt_num'] += 1
            variables['temperature'] *= (1 - 0.5 * 300 / args.number_iterations)
            if variables['SA_interrupt_num'] > len(variables['SA_schedule']):
                variables['SA_interrupt_num'] -= 1

        # Check for GA turn
        if iteration == variables['GA_schedule'][variables['GA_interrupt_num'] - 1]:
            variables['GA_fitness'] = variables['gene_fitness'] / iterations_until_GA + genetic_elite
            best_GA_fitness_candidate = np.max(variables['GA_fitness'])
            best_GA_index_candidate = np.argmax(variables['GA_fitness'])
            if best_GA_fitness_candidate >= variables['best_GA_fitness']:
                variables['best_GA_fitness'] = best_GA_fitness_candidate

                
                best_chromosome = [
                    variables['probability_factors']['up'][2 * best_GA_index_candidate],
                    variables['probability_factors']['right'][2 * best_GA_index_candidate],
                    variables['probability_factors']['alpha'][2 * best_GA_index_candidate],
                    variables['probability_factors']['beta'][2 * best_GA_index_candidate],
                    variables['probability_factors']['routing'][2 * best_GA_index_candidate]
                ]
            variables = GA(variables)
            variables['GA_interrupt_num'] += 1
            if variables['GA_interrupt_num'] > len(variables['GA_schedule']):
                variables['GA_interrupt_num'] -= 1
                # Assign best gene to all ants
                for idx, key in enumerate(['up', 'right', 'alpha', 'beta', 'routing']):
                    variables['probability_factors'][key] = variables['best_chromosome'][idx]

                # for key in ['up', 'right', 'alpha', 'beta', 'routing']:
                #     variables['probability_factors'][key][:] = variables['best_chromosome']

            iterations_until_GA = 0
            variables['gene_fitness'].fill(0)
            genetic_elite.fill(0)

        # Pheromone map update
        weighted_pheromone_trace_matrix = np.einsum('ijk,k->ijk', variables['pheromone_trace_matrix'], variables['fitness_vector'])
        pheromone_map = 0.6 * pheromone_map + np.sum(weighted_pheromone_trace_matrix, axis=2) / (30 * variables['best_fitness']) + 0.1 * variables['elite_pheromone_trace'] / 30

        # Process and calculate fitness for output
        best_fitness, im_enhanced = calculate_fitness(input_image, variables['enhancement_lut'], args.fitness_function)
        variables['fitness_per_iteration'][iteration - 1] = best_fitness

        im_enhanced = make_single_channel_image(im_enhanced)


        if iteration == args.number_iterations:  # Check if it's the last iteration
            print(f"Average pixel intensity: {np.mean(im_enhanced)}")
            sa_status = "SA_ON" if args.sa_enable.lower() == 'true' else "SA_OFF"
            # Extract the base filename without extension and prepend with 'Enhanced'
            base_filename = "Enhanced1_" + os.path.splitext(os.path.basename(args.image_path))[0]
            # Construct the LUT filename
            lut_filename = f"{base_filename}_{sa_status}_#It_{args.number_iterations}_LUT.npy"
            # Construct the figure filename
            figure_filename = f"{base_filename}_{sa_status}_#It_{args.number_iterations}.png"

            # Save LUT of enhanced image
            if args.save_lut.lower() == 'true':
                np.save(lut_filename, variables['enhancement_lut'])
                print(np.shape(variables['enhancement_lut']))
                print(f"\nLUT saved to {lut_filename}")

                        # Saving the pheromone map
            if args.save_pheromone_map.lower() == 'true':
                max_overlayed_data = np.amax(variables["pheromone_trace_matrix"], axis=2)
                # max_overlayed_data = variables["elite_pheromone_trace"] #elite trace

                capped_data = np.clip(max_overlayed_data, 0, 1)
                inverted_matrix = 1 - capped_data
                plt.imshow(inverted_matrix, cmap='gray', aspect='auto',
                    extent=[0, 255, 0, 255])

                plt.xlabel('Input image')
                plt.ylabel('Output image')
                plt.title('Pheromone path ants')
                plt.show()
                # plt.savefig(filename+'_pheromone_map.jpg')  # For a JPG filE

            # Save enhanced image
            cv2.imwrite(figure_filename, im_enhanced)
            print(f"\nEnhanced imaged saved to {figure_filename}")

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")


    # summed_pheromone_trace = np.sum(variables["pheromone_trace_matrix"], axis=2)





# Argument parser setup
if __name__ == '__main__':
    # Setup the argument parser
    parser = argparse.ArgumentParser(description='Process images with various options.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--fitness_function', type=str, default='brisque', choices=['brisque', 'classic'], help='Select the fitness function')
    parser.add_argument('--number_iterations', type=int, default=30, help='Number of iterations for processing')
    parser.add_argument('--sa_enable', type=str, choices=['True', 'False'], default='False', help='Enable simulated annealing by setting True or False')
    parser.add_argument('--save_lut', type=str, choices=['True', 'False'], default='False', help='Save the LUT after processing by setting True or False')
    parser.add_argument('--save_pheromone_map', type=str, choices=['True', 'False'], default='False', help='Save the pheromone map if applicable by setting True or False')

    args = parser.parse_args()
    main(args)
