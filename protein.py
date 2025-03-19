import random
import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def simulated_annealing_3d_folding_improved(sequence, grid_size=5, iterations=10000,
                                            initial_temp=1.0, cooling_rate=0.001, delay=0.01,
                                            step_size=1.0, contact_threshold=1.5):

    def calculate_energy(configuration, sequence):
        # Start with energy 0.
        energy = 0
        n = configuration.shape[0]
        # Only consider the 'H' residues (skip immediate neighbors).
        for i in range(n):
            if sequence[i] != 'H':
                continue
            diffs = configuration[i+2:] - configuration[i]
            dists = np.linalg.norm(diffs, axis=1)
            # For each 'H' residue that comes within the contact threshold, lower the energy.
            for j, dist in enumerate(dists, start=i+2):
                if dist <= contact_threshold and sequence[j] == 'H':
                    energy -= 1
        return energy

    def is_valid(configuration, index, new_coord):
        # Check if the new point stays inside our box.
        if not (0 <= new_coord[0] < grid_size and
                0 <= new_coord[1] < grid_size and
                0 <= new_coord[2] < grid_size):
            return False
        # Remove the point we're moving and check that the new point isn't too close to any others.
        temp_config = np.delete(configuration, index, axis=0)
        if np.any(np.linalg.norm(temp_config - new_coord, axis=1) < 0.1):
            return False
        return True

    def generate_random_start(sequence, grid_size):
        n = len(sequence)
        config = []
        # Choose random spots while keeping points at least 0.5 units apart.
        while len(config) < n:
            coord = np.array([random.uniform(0, grid_size) for _ in range(3)])
            if all(np.linalg.norm(coord - np.array(existing)) >= 0.5 for existing in config):
                config.append(coord)
        return np.array(config)

    def generate_neighbor(configuration, sequence, step_size, max_attempts=100):
        new_config = configuration.copy()
        n = new_config.shape[0]
        for attempt in range(max_attempts):
            index = random.randint(0, n - 1)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            new_coord = new_config[index] + direction * step_size
            if is_valid(new_config, index, new_coord):
                new_config[index] = new_coord
                return new_config
        return new_config

    # Start with a random configuration and get its energy.
    current_config = generate_random_start(sequence, grid_size)
    current_energy = calculate_energy(current_config, sequence)
    best_config = current_config.copy()
    best_energy = current_energy
    temp = initial_temp

    # Record energy changes over time.
    energy_history = []

    # Set up the 3D plot for the protein structure.
    fig_protein = plt.figure("Protein Structure")
    ax_protein = fig_protein.add_subplot(111, projection='3d')

    # Set up the energy vs. iteration plot.
    fig_energy = plt.figure("Energy Over Time")
    ax_energy = fig_energy.add_subplot(111)

    plt.ion()

    for step in range(iterations):
        new_config = generate_neighbor(current_config, sequence, step_size)
        new_energy = calculate_energy(new_config, sequence)
        delta_energy = new_energy - current_energy

        # Decide to accept the move if it lowers energy or sometimes even if it doesn't.
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / temp):
            current_config = new_config
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_config = current_config.copy()

        energy_history.append(current_energy)

        # Cool down the system and reduce the move size slightly.
        temp *= (1 - cooling_rate)
        step_size = max(0.01, step_size * 0.999)

        if step % 50 == 0:
            ax_protein.clear()
            ax_protein.plot(current_config[:, 0], current_config[:, 1], current_config[:, 2], marker='o')
            ax_protein.set_xlabel('X')
            ax_protein.set_ylabel('Y')
            ax_protein.set_zlabel('Z')
            ax_protein.set_xlim(0, grid_size)
            ax_protein.set_ylim(0, grid_size)
            ax_protein.set_zlim(0, grid_size)
            ax_protein.set_title(f"Protein at step {step}")

            ax_energy.clear()
            ax_energy.plot(energy_history, '-b')
            ax_energy.set_xlabel('Iteration')
            ax_energy.set_ylabel('Energy')
            ax_energy.set_title("Energy Evolution")

            plt.pause(delay)

    plt.ioff()
    plt.show()

    return best_energy, best_config

sequence = "PHPPHPHHP"
energy, configuration = simulated_annealing_3d_folding_improved(sequence)

if configuration.size:
    print("Best Energy:", energy)
    print("Best Configuration:\n", configuration)
else:
    print("No valid fold found.")
