import torch # Import the PyTorch library - FOR DEEP LEARNING
import torch.nn as nn # Import the nn module from PyTorch - FOR NEURAL NETWORK LAYERS
import torch.optim as optim # Import the optim module from PyTorch - FOR OPTIMIZERS
import torch.nn.functional as F # Import the F module from PyTorch - FOR ACTIVATION FUNCTIONS AND OTHER UTILITIES
import numpy as np # Import NumPy library - FOR NUMERICAL OPERATIONS
import random # Import random library - FOR RANDOM NUMBER GENERATION
import os # Import os library - FOR FILE SYSTEM OPERATIONS
from collections import deque # Import deque from collections - FOR EFFICIENT QUEUES

# ---------------- SumTree Class ---------------- #
class SumTree:
    """Data structure to store priorities for prioritized experience replay."""
    def __init__(self, capacity):
        """Initialize the SumTree with a given capacity."""
        self.capacity = capacity # Store the capacity - MAXIMUM NUMBER OF EXPERIENCES
        self.tree = np.zeros(2 * capacity - 1) # Initialize the tree with zeros - REPRESENTATION OF PRIORITIES
        self.data = np.zeros(capacity, dtype=object) # Initialize the data array - STORE EXPERIENCES
        self.size = 0 # Current number of elements in the tree
        self.write = 0 # Index to write new elements

    def _propagate(self, idx, change):
        """Propagate changes in priority up the tree."""
        parent = (idx - 1) // 2 # Calculate the parent index - GET THE PARENT NODE
        self.tree[parent] += change # Update the parent priority - UPDATE THE SUM
        if parent != 0: # If not root node, propagate further - CONTINUE UP THE TREE
            self._propagate(parent, change) # Recursively propagate - RECURSIVE CALL

    def _retrieve(self, idx, s):
        """Retrieve the index based on a sample value."""
        left = 2 * idx + 1 # Calculate the left child index - GET THE LEFT CHILD
        right = left + 1 # Calculate the right child index - GET THE RIGHT CHILD
        if left >= len(self.tree): # If leaf node, return the index - REACHED A LEAF
            return idx  # leaf node - RETURN THE LEAF INDEX
        if s <= self.tree[left]: # If sample value is less than the left child - CHECK THE LEFT SUBTREE
            return self._retrieve(left, s) # Recursively retrieve from the left child - RECURSIVE CALL
        else: # Otherwise, retrieve from the right child - CHECK THE RIGHT SUBTREE
            return self._retrieve(right, s - self.tree[left]) # Recursively retrieve from the right child - RECURSIVE CALL

    def total(self):
        """Return the total priority value."""
        return self.tree[0] # The root node contains the total priority - RETURN THE ROOT NODE VALUE

    def add(self, priority, data):
        """Add a new experience to the tree."""
        idx = self.write + self.capacity - 1 # Calculate the index to write - INDEX IN THE TREE
        self.data[self.write] = data # Store the data - STORE THE EXPERIENCE
        self.update(idx, priority) # Update the priority in the tree - UPDATE THE SUM TREE
        self.write = (self.write + 1) % self.capacity # Update the write index - CIRCULAR BUFFER
        self.size = min(self.size + 1, self.capacity) # Update the size of the tree - KEEP TRACK OF SIZE

    def update(self, idx, priority):
        """Update the priority of an experience in the tree."""
        change = priority - self.tree[idx] # Calculate the change in priority - PRIORITY DIFFERENCE
        self.tree[idx] = priority # Update the priority - SET THE NEW PRIORITY
        self._propagate(idx, change) # Propagate the change - UPDATE PARENT NODES

    def get(self, s):
        """Get the index, priority, and data based on a sample value."""
        idx = self._retrieve(0, s) # Retrieve the index - GET THE INDEX
        data_idx = idx - self.capacity + 1 # Calculate the data index - INDEX IN THE DATA ARRAY
        return idx, self.tree[idx], self.data[data_idx] # Return the index, priority, and data - RETURN THE VALUES

    def sample(self):
        """Sample an experience based on its priority."""
        s = random.uniform(0, self.total())  # Select a random value to sample - SAMPLE A RANDOM VALUE
        idx = self._retrieve(0, s) # Retrieve the index - GET THE INDEX BASED ON THE RANDOM VALUE
        data_idx = idx - self.capacity + 1  # Find the corresponding experience - CALCULATE THE DATA INDEX
        return idx, self.tree[idx], self.data[data_idx]  # Return the correct tuple - RETURN THE TUPLE

# ---------------- PER_ReplayBuffer Class ---------------- #
class PER_ReplayBuffer:
    """Replay buffer with Prioritized Experience Replay."""
    def __init__(self, capacity):
        """Initialize the PER replay buffer."""
        self.capacity = capacity # Store the capacity - MAXIMUM NUMBER OF EXPERIENCES
        self.tree = SumTree(capacity)  # Assuming you are using a SumTree for PER - SUM TREE DATA STRUCTURE
        self.buffer = []  # Buffer for storing experiences - STORE EXPERIENCES
        self.index = 0  # Index to keep track of where to store new experiences - INDEX FOR CIRCULAR BUFFER

    def add(self, priority, experience):
        """Add an experience to the replay buffer with a given priority."""
        if len(self.buffer) < self.capacity: # If the buffer is not full - CHECK IF THERE IS SPACE
            self.buffer.append(experience) # Append the experience - ADD TO THE BUFFER
        else: # If the buffer is full - REPLACE OLDEST EXPERIENCE
            self.buffer[self.index] = experience # Replace the experience at the current index - REPLACE EXISTING EXPERIENCE

        # Add experience to the SumTree - UPDATE THE SUM TREE
        self.tree.add(priority, experience) # Add the priority and experience to the sum tree - ADD TO THE SUM TREE

        # Update index - UPDATE THE INDEX
        self.index = (self.index + 1) % self.capacity # Increment the index - CIRCULAR BUFFER

    def sample(self, batch_size):
        """Sample a batch from the replay buffer."""
        batch = [] # List to store the sampled batch - STORE THE BATCH
        idxs = [] # List to store the indices of the sampled experiences - STORE THE INDICES
        weights = [] # List to store the weights of the sampled experiences - STORE THE WEIGHTS

        # Sample experiences from the SumTree with proportional prioritization - SAMPLE FROM THE SUM TREE
        for _ in range(batch_size): # Repeat for the batch size - ITERATE TO CREATE THE BATCH
            idx, priority, experience = self.tree.sample() # Sample an experience - SAMPLE AN EXPERIENCE
            batch.append(experience) # Append the experience to the batch - ADD TO THE BATCH
            idxs.append(idx) # Append the index to the list of indices - ADD THE INDEX
            weights.append(self._get_weight(priority)) # Append the weight to the list of weights - ADD THE WEIGHT

        # Convert weights to a tensor - CONVERT TO TENSOR
        weights = torch.tensor(weights, dtype=torch.float32) # Convert to a PyTorch tensor - CONVERT TO TENSOR
        return batch, idxs, weights # Return the batch, indices, and weights - RETURN THE VALUES

    def update(self, idx, priority):
        """Update the priority of an experience in the SumTree."""
        self.tree.update(idx, priority) # Update the priority in the SumTree - UPDATE THE PRIORITY IN THE TREE

    def _get_weight(self, priority):
        """Calculate the weight for an experience based on its priority."""
        # This is a simple placeholder; you might need to tweak this based on your algorithm. - ADJUST BASED ON ALGORITHM
        return priority # Return the priority - RETURN THE PRIORITY

    def __len__(self):
        """Returns the number of experiences in the buffer."""
        return len(self.buffer) # Return the length of the buffer - RETURN THE SIZE

# ---------------- NoisyLinear Class ---------------- #
class NoisyLinear(nn.Module):
    """Noisy layer for exploration."""
    def __init__(self, in_features, out_features, std_init=0.5):
        """Initialize the NoisyLinear layer."""
        super(NoisyLinear, self).__init__() # Call the parent class constructor - INITIALIZE THE PARENT CLASS
        self.in_features = in_features # Store the input features - DIMENSION OF THE INPUT
        self.out_features = out_features # Store the output features - DIMENSION OF THE OUTPUT
        self.std_init = std_init # Store the standard deviation - INITIAL STANDARD DEVIATION
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features)) # Initialize the weight mu - MEAN OF THE WEIGHTS
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features)) # Initialize the weight sigma - STANDARD DEVIATION OF THE WEIGHTS
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features)) # Register the weight epsilon - NOISE FOR THE WEIGHTS
        
        self.bias_mu = nn.Parameter(torch.empty(out_features)) # Initialize the bias mu - MEAN OF THE BIAS
        self.bias_sigma = nn.Parameter(torch.empty(out_features)) # Initialize the bias sigma - STANDARD DEVIATION OF THE BIAS
        self.register_buffer("bias_epsilon", torch.empty(out_features)) # Register the bias epsilon - NOISE FOR THE BIAS
        
        self.reset_parameters() # Reset the parameters - INITIALIZE THE PARAMETERS
        self.reset_noise() # Reset the noise - INITIALIZE THE NOISE
    
    def reset_parameters(self):
        """Reset the parameters of the layer."""
        mu_range = 1 / np.sqrt(self.in_features) # Calculate the range for mu - CALCULATE THE RANGE
        self.weight_mu.data.uniform_(-mu_range, mu_range) # Initialize the weight mu with a uniform distribution - INITIALIZE THE WEIGHT MEAN
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features)) # Initialize the weight sigma with a fixed value - INITIALIZE THE WEIGHT STANDARD DEVIATION
        self.bias_mu.data.uniform_(-mu_range, mu_range) # Initialize the bias mu with a uniform distribution - INITIALIZE THE BIAS MEAN
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.in_features)) # Initialize the bias sigma with a fixed value - INITIALIZE THE BIAS STANDARD DEVIATION
    
    def reset_noise(self):
        """Reset the noise of the layer."""
        self.weight_epsilon.normal_() # Initialize the weight epsilon with a normal distribution - GENERATE WEIGHT NOISE
        self.bias_epsilon.normal_() # Initialize the bias epsilon with a normal distribution - GENERATE BIAS NOISE
    
    def forward(self, x):
        """Forward pass of the layer."""
        return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, # Calculate the output - LINEAR TRANSFORMATION WITH NOISE
                        self.bias_mu + self.bias_sigma * self.bias_epsilon) # Add the bias - ADD NOISY BIAS

# ---------------- CNN_DQN Class ---------------- #
class CNN_DQN(nn.Module):
    """CNN-based DQN model."""
    def __init__(self, input_shape, num_actions):
        """Initialize the CNN_DQN model."""
        super(CNN_DQN, self).__init__() # Call the parent class constructor - INITIALIZE THE PARENT CLASS

        # Convert input shape to PyTorch format (C, H, W) - CONVERT TO CHANNEL, HEIGHT, WIDTH
        self.input_shape = (input_shape[2], input_shape[0], input_shape[1])  # STORE THE INPUT SHAPE
        self.num_actions = num_actions # STORE THE NUMBER OF ACTIONS

        self.conv_layers = nn.Sequential( # DEFINE THE CONVOLUTIONAL LAYERS
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4), # First convolutional layer - CONVOLUTIONAL LAYER
            nn.ReLU(), # ReLU activation - RELU ACTIVATION
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Second convolutional layer - CONVOLUTIONAL LAYER
            nn.ReLU(), # ReLU activation - RELU ACTIVATION
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Third convolutional layer - CONVOLUTIONAL LAYER
            nn.ReLU() # ReLU activation - RELU ACTIVATION
        )

        self.fc_input_size = self._get_conv_out(self.input_shape)  # Compute dynamically - CALCULATE THE OUTPUT SIZE OF THE CONVOLUTIONAL LAYERS

        self.fc_layers = nn.Sequential( # DEFINE THE FULLY CONNECTED LAYERS
            NoisyLinear(self.fc_input_size, 512), # First fully connected layer - NOISY LINEAR LAYER
            nn.ReLU(), # ReLU activation - RELU ACTIVATION
            NoisyLinear(512, num_actions) # Second fully connected layer - NOISY LINEAR LAYER
        )

    def _get_conv_out(self, shape):
        """Compute the size of the flattened convolutional output."""
        with torch.no_grad(): # Disable gradient calculation - AVOID GRADIENT CALCULATION
            dummy_input = torch.zeros(1, *shape)  # (batch, channels, height, width) - CREATE A DUMMY INPUT
            conv_out = self.conv_layers(dummy_input) # Pass the dummy input through the convolutional layers - GET THE OUTPUT
            return int(np.prod(conv_out.shape[1:]))  # Flattened size - CALCULATE THE FLATTENED SIZE

    def forward(self, x):
        """Forward pass of the CNN_DQN model."""
        x = self.conv_layers(x) # Pass the input through the convolutional layers - CONVOLUTIONAL LAYERS
        x = torch.flatten(x, start_dim=1)  # Ensure correct flattening - FLATTEN THE OUTPUT
        return self.fc_layers(x) # Pass the flattened output through the fully connected layers - FULLY CONNECTED LAYERS

# ---------------- CNN_DQN_Agent Class ---------------- #
class CNN_DQN_Agent:
    """CNN-DQN agent with Prioritized Experience Replay."""
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, batch_size=32, model_path=None, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """Initialize the CNN_DQN_Agent."""
        self.input_shape = input_shape # Store the input shape - INPUT DIMENSIONS
        self.num_actions = num_actions # Store the number of actions - NUMBER OF POSSIBLE ACTIONS
        self.gamma = gamma # Store the discount factor - DISCOUNT FUTURE REWARDS
        self.batch_size = batch_size # Store the batch size - TRAINING BATCH SIZE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine the device - USE GPU IF AVAILABLE
        
        # Model and target model for DQN - CREATE THE MODELS
        self.model = CNN_DQN(input_shape, num_actions).to(self.device) # Create the main model - MAIN Q-NETWORK
        self.target_model = CNN_DQN(input_shape, num_actions).to(self.device) # Create the target model - TARGET Q-NETWORK
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # Create the optimizer - ADAM OPTIMIZER
        self.replay_buffer = PER_ReplayBuffer(100000) # Create the replay buffer - PER REPLAY BUFFER

        self.model_path = model_path # STORE THE MODEL PATH
        if model_path and os.path.exists(model_path): # CHECK IF THE MODEL EXISTS
            self.load(model_path) # Load the model - LOAD THE TRAINED WEIGHTS

        # Epsilon for epsilon-greedy policy - EPSILON GREEDY
        self.epsilon = epsilon_start # Store the initial epsilon value - INITIAL EXPLORATION RATE
        self.epsilon_end = epsilon_end # Store the final epsilon value - MINIMUM EXPLORATION RATE
        self.epsilon_decay = epsilon_decay # Store the epsilon decay value - EXPLORATION DECAY RATE

        # Copy weights from the model to the target model - INITIALIZE TARGET NETWORK
        self.update_target_model() # Update the target model - SYNCHRONIZE TARGET NETWORK

    def get_action(self, state):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon: # CHECK IF WE SHOULD EXPLORE - RANDOM ACTION
            return np.random.randint(self.num_actions)  # Random action (exploration) - EXPLORE
        
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device) # Convert the state to a tensor - CONVERT TO TENSOR
        
        with torch.no_grad(): # Disable gradient calculation - DISABLE GRADIENTS FOR INFERENCE
            q_values = self.model(state) # Get the Q-values for the state - GET THE Q-VALUES
        return torch.argmax(q_values).item()  # Greedy action (exploitation) - EXPLOIT - CHOOSE THE BEST ACTION

    def update_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) # Decay epsilon - UPDATE THE EXPLORATION RATE

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the PER buffer."""
        error = abs(reward)  # Placeholder error - INITIAL ERROR (CAN BE TD ERROR)
        self.replay_buffer.add(error, (state, action, reward, next_state, done)) # Add the experience to the replay buffer - STORE THE EXPERIENCE

    def learn(self):
        """Update the model using experiences sampled from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size: # CHECK IF THERE IS ENOUGH DATA - NEED ENOUGH SAMPLES
            return # Not enough data to train - WAIT UNTIL THERE IS ENOUGH DATA

        # Sample a batch of experiences - SAMPLE A BATCH
        batch, idxs, weights = self.replay_buffer.sample(self.batch_size) # Sample from the replay buffer - GET THE SAMPLES

        # Convert list of numpy arrays to a single numpy array before creating a tensor - CONVERT TO NUMPY ARRAY
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device) # Convert states to a tensor - CONVERT TO TENSOR
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device) # Convert actions to a tensor - CONVERT TO TENSOR
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(self.device) # Convert rewards to a tensor - CONVERT TO TENSOR
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device) # Convert next states to a tensor - CONVERT TO TENSOR
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(self.device) # Convert dones to a tensor - CONVERT TO TENSOR

        # Compute Q values for current states - CALCULATE THE Q-VALUES
        q_values = self.model(states) # Get the Q-values for the current states - FOR THE CURRENT STATES
        q_value = q_values.gather(1, actions.unsqueeze(1)) # Get the Q-value for the taken actions - FOR THE CHOSEN ACTIONS

        # Compute Q values for next states - CALCULATE THE Q-VALUES FOR THE NEXT STATES
        next_q_values = self.target_model(next_states) # Get the Q-values for the next states - USING THE TARGET NETWORK
        next_q_value = next_q_values.max(1)[0] # Get the maximum Q-value - GET THE MAXIMUM VALUE

        # Compute target: reward + gamma * max(Q_next) * (1 - done) - CALCULATE THE TARGET
        target = rewards + self.gamma * next_q_value * (1 - dones) # Calculate the target Q-value - Q-LEARNING TARGET

        # Compute the loss (TD error) - CALCULATE THE LOSS
        loss = (weights * F.mse_loss(q_value.squeeze(), target, reduction='none')).mean() # Calculate the loss - MEAN SQUARED ERROR

        # Backpropagate and optimize - TRAIN THE MODEL
        self.optimizer.zero_grad() # Reset gradients - CLEAR THE GRADIENTS
        loss.backward() # Backpropagate the loss - CALCULATE THE GRADIENTS
        self.optimizer.step() # Update the weights - UPDATE THE WEIGHTS

        # Update priorities in the buffer - UPDATE PRIORITIES IN THE REPLAY BUFFER
        for i, idx in enumerate(idxs): # Iterate over the indices - ITERATE OVER THE BATCH
            self.replay_buffer.update(idx, abs(target[i] - q_value[i].item())) # Update the priority - BASED ON THE TD ERROR

        # Update epsilon - UPDATE THE EXPLORATION RATE
        self.update_epsilon() # Update epsilon - DECAY EPSILON

    def update_target_model(self):
        """Update the target model with the current model weights."""
        self.target_model.load_state_dict(self.model.state_dict()) # Load the state dictionary - COPY THE WEIGHTS

    def load(self, model_path):
        """Load the model from the specified path."""
        self.model.load_state_dict(torch.load(model_path)) # Load the state dictionary - LOAD THE WEIGHTS
        self.update_target_model() # Update the target model - SYNCHRONIZE THE TARGET NETWORK

    def save(self, model_path):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), model_path) # Save the state dictionary - SAVE THE WEIGHTS
