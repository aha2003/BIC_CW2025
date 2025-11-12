'''
import numpy as np

class PSO:
    """
    Particle Swarm Optimization (PSO) implementation based on the specific
    "Algorithm 39" from "Essentials of Metaheuristics".
    
    This version uses informants and can include a global best component,
    and follows the explicit structure laid out in the pseudocode.
    """

    def __init__(self, objective_function, dim, swarm_size, num_iterations,
                 w=0.7, c1=1.5, c2=1.5, num_informants=5, param_bounds=(-1, 1)):
        """
        Initializes the PSO optimizer.
        Args:
            objective_function (callable): The function to minimize.
            dim (int): The number of dimensions of the problem.
            swarm_size (int): The number of particles in the swarm. (Pseudocode: swarmsize)
            num_iterations (int): The number of optimization iterations.
            w (float): Inertia weight. (Pseudocode: alpha)
            c1 (float): Cognitive coefficient. (Pseudocode: beta)
            c2 (float): Social (informant) coefficient. (Pseudocode: gamma)
            num_informants (int): The number of informants for each particle.
            param_bounds (tuple): A tuple (min, max) for particle position initialization.
        """
        self.objective_function = objective_function
        self.dim = dim
        self.swarm_size = swarm_size # Line 1: swarmsize
        self.num_iterations = num_iterations

        # --- Map parameters to Algorithm 39 pseudocode ---
        self.alpha = w      # Line 2: alpha
        self.beta = c1      # Line 3: beta
        self.gamma = c2     # Line 4: gamma
        self.delta = 0.0    # Line 5: delta (set to 0 as is common in modern PSO)
        self.e = 1.0        # Line 6: e (jump size, typically 1)
        # ------------------------------------------------

        self.num_informants = num_informants
        self.min_bound, self.max_bound = param_bounds

        # --- PSO Algorithm Initialization ---
        
        # Line 7: P <- {}
        # Line 8-9: Create 'swarmsize' random particles with random initial velocities
        self.positions = np.random.uniform(self.min_bound, self.max_bound, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        
        # Initialize personal best positions (x*) and their fitness values
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness = np.full(self.swarm_size, np.inf)

        # Initialize informants for each particle (needed for x+)
        self.informants = self._initialize_informants()
        
        # Line 10: Best <- NULL
        # Initialize global best position (x!) and its fitness value
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def _initialize_informants(self):
        """
        Initializes the informant set for each particle using a ring topology.
        This is a setup step required for finding x+ (Line 18) later.
        """
        informants = {}
        for i in range(self.swarm_size):
            indices = [(i + j) % self.swarm_size for j in range(1, self.num_informants + 1)]
            informants[i] = indices
        return informants

    def optimize(self):
        """
        Runs the PSO optimization loop according to Algorithm 39.
        """
        # Line 11: repeat (main optimization loop)
        for t in range(self.num_iterations):
            
            # --- Part 1: Assess Fitness and Update Bests ---
            # This block corresponds to the first loop in the pseudocode (lines 12-15),
            # where fitness is assessed and the global best is updated.
            for i in range(self.swarm_size):
                # Line 13: AssessFitness(x)
                current_fitness = self.objective_function(self.positions[i])

                # Update personal best (x* for the next iteration)
                # This happens implicitly as we check against the previous pbest
                if current_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness
                    self.pbest_positions[i] = np.copy(self.positions[i])
                
                # Line 14: if Best is NULL or Fitness(x) > Fitness(Best)
                # Note: We use '<' for minimization, while pseudocode uses '>' for maximization.
                if self.global_best_position is None or current_fitness < self.global_best_fitness:
                    # Line 15: Best <- x
                    self.global_best_fitness = current_fitness
                    self.global_best_position = np.copy(self.positions[i])

            # --- Part 2: Determine and Apply Mutation (Update Velocity and Position) ---
            # This block corresponds to the second and third loops in the pseudocode (lines 16-26)
            for i in range(self.swarm_size):
                # --- Determine how to Mutate (Velocity Update) ---
                # Line 17: x* <- previous fittest location of x (personal best)
                pbest = self.pbest_positions[i]
                
                # Line 18: x+ <- previous fittest location of informants of x
                informant_indices = self.informants[i]
                best_informant_idx = min(informant_indices, key=lambda j: self.pbest_fitness[j])
                nbest = self.pbest_positions[best_informant_idx]
                
                # Line 19: x! <- previous fittest location any particle (global best)
                gbest = self.global_best_position
                
                # Lines 20-24: Update velocity for each dimension i
                # This is vectorized for efficiency instead of a for-loop over dimensions.
                
                # Line 21: b <- random number from 0.0 to beta inclusive
                b = np.random.uniform(0, self.beta, self.dim)
                # Line 22: c <- random number from 0.0 to gamma inclusive
                c = np.random.uniform(0, self.gamma, self.dim)
                # Line 23: d <- random number from 0.0 to delta inclusive
                d = np.random.uniform(0, self.delta, self.dim)
                
                # Line 24: vi <- alpha*vi + b*(x*_i - xi) + c*(x+_i - xi) + d*(x!_i - xi)
                cognitive_component = b * (pbest - self.positions[i])
                social_component = c * (nbest - self.positions[i])
                global_component = d * (gbest - self.positions[i])
                
                self.velocities[i] = self.alpha * self.velocities[i] + cognitive_component + social_component + global_component
            
            # --- Mutate (Position Update) ---
            # Line 25-26: For each particle, update position: x <- x + e*v
            self.positions = self.positions + self.e * self.velocities
            
            # Boundary Handling (An extension to the core algorithm for stability) 
            self.positions = np.clip(self.positions, self.min_bound, self.max_bound)

            #print(f"Iteration {t+1}/{self.num_iterations}, Best Fitness Found: {self.global_best_fitness:.4f}")

        # Line 27: until Best is ideal or out of time (end of loop)
        
        # Line 28: return Best
        return self.global_best_position, self.global_best_fitness
 '''
# pso.py (Modified for selectable boundary handling)
import numpy as np

class PSO:
    """
    Particle Swarm Optimization (PSO) implementation based on the specific
    "Algorithm 39" from "Essentials of Metaheuristics".
    
    This version uses informants and can include a global best component,
    and follows the explicit structure laid out in the pseudocode.
    
    MODIFIED: Includes selectable boundary handling strategies.
    """

    def __init__(self, objective_function, dim, swarm_size, num_iterations,
                 w=0.7, c1=1.5, c2=1.5, num_informants=5, param_bounds=(-1, 1),
                 boundary_strategy='clamp'): # <--- NEW PARAMETER
        """
        Initializes the PSO optimizer.
        Args:
            objective_function (callable): The function to minimize.
            dim (int): The number of dimensions of the problem.
            swarm_size (int): The number of particles in the swarm. (Pseudocode: swarmsize)
            num_iterations (int): The number of optimization iterations.
            w (float): Inertia weight. (Pseudocode: alpha)
            c1 (float): Cognitive coefficient. (Pseudocode: beta)
            c2 (float): Social (informant) coefficient. (Pseudocode: gamma)
            num_informants (int): The number of informants for each particle.
            param_bounds (tuple): A tuple (min, max) for particle position initialization.
            boundary_strategy (str): 'clamp', 'reflect', or 'random'. # <--- NEW
        """
        self.objective_function = objective_function
        self.dim = dim
        self.swarm_size = swarm_size # Line 1: swarmsize
        self.num_iterations = num_iterations

        # --- Map parameters to Algorithm 39 pseudocode ---
        self.alpha = w      # Line 2: alpha
        self.beta = c1      # Line 3: beta
        self.gamma = c2     # Line 4: gamma
        self.delta = 0.0    # Line 5: delta (set to 0 as is common in modern PSO)
        self.e = 1.0        # Line 6: e (jump size, typically 1)
        # ------------------------------------------------

        self.num_informants = num_informants
        self.min_bound, self.max_bound = param_bounds

        # --- NEW: Store boundary strategy ---
        self.boundary_strategy = boundary_strategy
        if self.boundary_strategy not in ['clamp', 'reflect', 'random']:
            raise ValueError("boundary_strategy must be one of 'clamp', 'reflect', or 'random'")
        # ------------------------------------

        # --- PSO Algorithm Initialization ---
        
        # Line 7: P <- {}
        # Line 8-9: Create 'swarmsize' random particles with random initial velocities
        self.positions = np.random.uniform(self.min_bound, self.max_bound, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        
        # Initialize personal best positions (x*) and their fitness values
        self.pbest_positions = np.copy(self.positions)
        self.pbest_fitness = np.full(self.swarm_size, np.inf)

        # Initialize informants for each particle (needed for x+)
        self.informants = self._initialize_informants()
        
        # Line 10: Best <- NULL
        # Initialize global best position (x!) and its fitness value
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def _initialize_informants(self):
        """
        Initializes the informant set for each particle using a ring topology.
        This is a setup step required for finding x+ (Line 18) later.
        """
        informants = {}
        for i in range(self.swarm_size):
            indices = [(i + j) % self.swarm_size for j in range(1, self.num_informants + 1)]
            informants[i] = indices
        return informants

    # <--- NEW METHOD ---
    def _apply_boundary_handling(self):
        """Applies the selected boundary handling strategy to particle positions."""
        
        if self.boundary_strategy == 'clamp':
            # Strategy 1: Clamp position to the boundary
            # This was the original behavior
            self.positions = np.clip(self.positions, self.min_bound, self.max_bound)
            # Velocity is not changed
            
        elif self.boundary_strategy == 'reflect':
            # Strategy 2: Clamp position and reverse velocity
            over_mask = self.positions > self.max_bound
            under_mask = self.positions < self.min_bound
            
            # Clamp positions
            self.positions[over_mask] = self.max_bound
            self.positions[under_mask] = self.min_bound
            
            # Reverse velocity for the dimensions that were out of bounds
            self.velocities[over_mask] *= -1
            self.velocities[under_mask] *= -1

        elif self.boundary_strategy == 'random':
            # Strategy 3: Re-initialize out-of-bounds positions
            
            # Create a mask for all particles (rows) that have at least one dimension out of bounds
            out_of_bounds_mask = np.any((self.positions > self.max_bound) | (self.positions < self.min_bound), axis=1)
            
            num_out_of_bounds = np.sum(out_of_bounds_mask)
            
            if num_out_of_bounds > 0:
                # Generate new random positions for *only* those particles
                new_positions = np.random.uniform(self.min_bound, self.max_bound, (num_out_of_bounds, self.dim))
                # Replace the old positions
                self.positions[out_of_bounds_mask] = new_positions
                # Optionally, reset velocity for these particles as well
                self.velocities[out_of_bounds_mask] = 0

    def optimize(self):
        """
        Runs the PSO optimization loop according to Algorithm 39.
        """
        # Line 11: repeat (main optimization loop)
        for t in range(self.num_iterations):
            
            # --- Part 1: Assess Fitness and Update Bests ---
            # This block corresponds to the first loop in the pseudocode (lines 12-15),
            # where fitness is assessed and the global best is updated.
            for i in range(self.swarm_size):
                # Line 13: AssessFitness(x)
                current_fitness = self.objective_function(self.positions[i])

                # Update personal best (x* for the next iteration)
                # This happens implicitly as we check against the previous pbest
                if current_fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = current_fitness
                    self.pbest_positions[i] = np.copy(self.positions[i])
                
                # Line 14: if Best is NULL or Fitness(x) > Fitness(Best)
                # Note: We use '<' for minimization, while pseudocode uses '>' for maximization.
                if self.global_best_position is None or current_fitness < self.global_best_fitness:
                    # Line 15: Best <- x
                    self.global_best_fitness = current_fitness
                    self.global_best_position = np.copy(self.positions[i])

            # --- Part 2: Determine and Apply Mutation (Update Velocity and Position) ---
            # This block corresponds to the second and third loops in the pseudocode (lines 16-26)
            for i in range(self.swarm_size):
                # --- Determine how to Mutate (Velocity Update) ---
                # Line 17: x* <- previous fittest location of x (personal best)
                pbest = self.pbest_positions[i]
                
                # Line 18: x+ <- previous fittest location of informants of x
                informant_indices = self.informants[i]
                best_informant_idx = min(informant_indices, key=lambda j: self.pbest_fitness[j])
                nbest = self.pbest_positions[best_informant_idx]
                
                # Line 19: x! <- previous fittest location any particle (global best)
                gbest = self.global_best_position
                
                # Lines 20-24: Update velocity for each dimension i
                # This is vectorized for efficiency instead of a for-loop over dimensions.
                
                # Line 21: b <- random number from 0.0 to beta inclusive
                b = np.random.uniform(0, self.beta, self.dim)
                # Line 22: c <- random number from 0.0 to gamma inclusive
                c = np.random.uniform(0, self.gamma, self.dim)
                # Line 23: d <- random number from 0.0 to delta inclusive
                d = np.random.uniform(0, self.delta, self.dim)
                
                # Line 24: vi <- alpha*vi + b*(x*_i - xi) + c*(x+_i - xi) + d*(x!_i - xi)
                cognitive_component = b * (pbest - self.positions[i])
                social_component = c * (nbest - self.positions[i])
                global_component = d * (gbest - self.positions[i])
                
                self.velocities[i] = self.alpha * self.velocities[i] + cognitive_component + social_component + global_component
            
            # --- Mutate (Position Update) ---
            # Line 25-26: For each particle, update position: x <- x + e*v
            self.positions = self.positions + self.e * self.velocities
            
            # Boundary Handling (An extension to the core algorithm for stability) 
            # self.positions = np.clip(self.positions, self.min_bound, self.max_bound) # <--- OLD
            self._apply_boundary_handling() # <--- NEW

            #print(f"Iteration {t+1}/{self.num_iterations}, Best Fitness Found: {self.global_best_fitness:.4f}")

        # Line 27: until Best is ideal or out of time (end of loop)
        
        # Line 28: return Best
        return self.global_best_position, self.global_best_fitness