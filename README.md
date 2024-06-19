# Natural Algorithm Inspired Contrast Enhancement

This repository contains the implementation of image contrast enhancement techniques using a hybrid approach that integrates Ant Colony Optimization (ACO), Genetic Algorithm (GA), and Simulated Annealing (SA). The algorithms have been designed to improve the visual quality of images by enhancing their contrast. This work is based on the work of Hoseini et al. 2013, "Efficient contrast enhancement of images using hybrid ant colony optimisation, genetic algorithm, and simulated annealing'

## Overview

Image contrast enhancement is a crucial process in various image processing applications. This project leverages the strengths of ACO, GA, and SA to achieve efficient and effective contrast enhancement. The hybrid approach combines the exploration capabilities of ACO and GA with the exploitation ability of SA, resulting in superior performance compared to using these algorithms independently.

## Installation

To run the code in this repository, you need to have Python installed along with the following libraries:
- numpy
- opencv-python
- matplotlib
- tqdm
- scikit-image
- brisque

You can install these libraries using pip:

pip install numpy opencv-python matplotlib tqdm scikit-image brisque

## Usage

python enhance_contrast.py path/to/your/image.jpg --fitness_function brisque --number_iterations 30 --sa_enable True --save_lut True --save_pheromone_map True

Arguments:
- image_path: Path to the image file.
- fitness_function: Select the fitness function ('brisque' or 'classic').
- number_iterations: Number of iterations for processing (default is 30).
- sa_enable: Enable simulated annealing ('True' or 'False', default is 'False').
- save_lut: Save the LUT after processing ('True' or 'False', default is 'False').
- save_pheromone_map: Save the pheromone map if applicable ('True' or 'False', default is 'False').

Citing This Work
If you use this code or any part of it in your research, please cite the following papers:
- Hoseini, P., & Shayesteh, M. G. (2013). Efficient contrast enhancement of images using hybrid ant colony optimisation, genetic algorithm, and simulated annealing. Digital Signal Processing, 23(3), 879-893. Publisher: Elsevier.
- Hoseini, P., & Shayesteh, M. G. (2010). Hybrid ant colony optimization, genetic algorithm, and simulated annealing for image contrast enhancement. In IEEE Congress on Evolutionary Computation (pp. 1-6). Organization: IEEE.

