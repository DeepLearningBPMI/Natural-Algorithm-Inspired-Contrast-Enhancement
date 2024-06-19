# Natural Algorithm Inspired Contrast Enhancement

This repository contains the implementation of image contrast enhancement techniques using a hybrid approach that integrates Ant Colony Optimization (ACO), Genetic Algorithm (GA), and Simulated Annealing (SA). The algorithms have been designed to improve the visual quality of images by enhancing their contrast.

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
@article{hoseini2013efficient,
  title={Efficient contrast enhancement of images using hybrid ant colony optimisation, genetic algorithm, and simulated annealing},
  author={Hoseini, Pourya and Shayesteh, Mahrokh G},
  journal={Digital Signal Processing},
  volume={23},
  number={3},
  pages={879--893},
  year={2013},
  publisher={Elsevier}
}

@inproceedings{hoseini2010hybrid,
  title={Hybrid ant colony optimization, genetic algorithm, and simulated annealing for image contrast enhancement},
  author={Hoseini, Pourya and Shayesteh, Mahrokh G},
  booktitle={IEEE Congress on Evolutionary Computation},
  pages={1--6},
  year={2010},
  organization={IEEE}
}

