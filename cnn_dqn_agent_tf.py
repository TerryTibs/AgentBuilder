import tensorflow as tf # Import TensorFlow library - FOR DEEP LEARNING
from tensorflow.keras import layers, models, optimizers, initializers # Import Keras modules - FOR BUILDING NEURAL NETWORKS
import numpy as np # Import NumPy library - FOR NUMERICAL OPERATIONS
import random # Import random library - FOR RANDOM NUMBER GENERATION
import os # Import os library - FOR FILE SYSTEM OPERATIONS
from collections import deque # Import deque from collections - FOR EFFICIENT QUEUES

# Check TensorFlow version and enable eager execution if needed - ENSURE TF IS WORKING
print("TensorFlow version:", tf.__version__) # PRINT THE TF VERSION
#tf.config.run_functions_eagerly(True) #For debugging (slows execution) - EAGER EXECUTION (FOR DEBUGGING)
#tf.experimental.numpy.experimental_enable_numpy_behavior()  #Uncomment if you get 'reshape' error - ENABLE EXPERIMENTAL NUMPY BEHAVIOR (IF NEEDED)

# ---------------- SumTree Class ---------------- #
class SumTree:
    """Data structure to store priorities for prioritized experience replay."""
    def __init__(self, capacity):
        """Initialize the SumTree with a given capacity."""
        self.capacity = capacity # Store the capacity - MAXIMUM SIZE
        self.tree = np.zeros(2 * capacity - 1) # Initialize the tree with zeros - THE TREE DATA STRUCTURE
        self.data = np.zeros(capacity, dtype=object) # Initialize the data array - STORE THE EXPERIENCES
        self.size = 0 # Current number of elements in the tree - CURRENT SIZE
        self.write = 0 # Index to write new elements - WRITE INDEX

    def _propagate(self, idx, change):
        """Propagate changes in priority up the tree."""
        parent = (idx - 1) // 2 # Calculate the parent index - PARENT NODE
        self.tree[parent] += change # Update the parent priority - UPDATE THE PARENT
        if parent != 0: # If not root node, propagate further - RECURSIVE CALL
            self._propagate(parent, change) # Recursively propagate - RECURSIVE PROPAGATION

    def _retrieve(self, idx, s):
        """Retrieve the index based on a sample value."""
        left = 2 * idx + 1 # Calculate the left child index - LEFT CHILD
        right = left + 1 # Calculate the right child index - RIGHT CHILD
        if left >= len(self.tree): # If leaf node, return the index - REACHED A LEAF
            return idx  # leaf node - RETURN THE LEAF INDEX
        if s <= self.tree[left]: # If sample value is less than the left child - CHECK THE LEFT SUBTREE
            return self._retrieve(left, s) # Recursively retrieve from the left child - RECURSIVE CALL
        else: # Otherwise, retrieve from the right child - CHECK THE RIGHT SUBTREE
            return self._retrieve(right, s - self.tree[left]) # Recursively retrieve from the right child - RECURSIVE CALL

    def total(self):
        """Return the total priority value."""
        return self.tree[0] # The root node contains the total priority - ROOT NODE

    def add(self, priority, data):
        """Add a new experience to the tree."""
        idx = self.write + self.capacity - 1 # Calculate the index to write - WRITE INDEX
        self.data[self.write] = data # Store the data - STORE THE DATA
        self.update(idx, priority) # Update the priority in the tree - UPDATE THE TREE
        self.write = (self.write + 1) % self.capacity # Update the write index - CIRCULAR BUFFER
        self.size = min(self.size + 1, self.capacity) # Update the size of the tree - UPDATE THE SIZE

    def update(self, idx, priority):
        """Update the priority of an experience in the tree."""
        change = priority - self.tree[idx] # Calculate the change in priority - CALCULATE THE CHANGE
        self.tree[idx] = priority # Update the priority - UPDATE THE VALUE
        self._propagate(idx, change) # Propagate the change - PROPAGATE THE CHANGE

    def get(self, s):
        """Get the index, priority, and data based on a sample value."""
        idx = self._retrieve(0, s) # Retrieve the index - GET THE INDEX
        data_idx = idx - self.capacity + 1 # Calculate the data index - DATA INDEX
        return idx, self.tree[idx], self.data[data_idx] # Return the index, priority, and data - RETURN THE VALUES

    def sample(self):
        """Sample an experience based on its priority."""
        s = random.uniform(0, self.total())  # Select a random value to sample - RANDOM NUMBER
        idx = self._retrieve(0, s) # Retrieve the index - GET THE INDEX
        data_idx = idx - self.capacity + 1  # Find the corresponding experience - DATA INDEX
        return idx, self.tree[idx], self.data[data_idx]  # Return the correct tuple - RETURN THE VALUES

# ---------------- PER_ReplayBuffer Class ---------------- #
class PER_ReplayBuffer:
    """Replay buffer with Prioritized Experience Replay."""
    def __init__(self, capacity):
        """Initialize the PER replay buffer."""
        self.capacity = capacity # Store the capacity - MAXIMUM SIZE
        self.tree = SumTree(capacity) # Initialize the SumTree - CREATE THE SUM TREE
        self.buffer = [] # Initialize the buffer - CREATE THE BUFFER
        self.index = 0 # Initialize the index - STARTING INDEX

    def add(self, priority, experience):
        """Add an experience to the replay buffer with a given priority."""
        if len(self.buffer) < self.capacity: # Check if the buffer is not full - CHECK THE SIZE
            self.buffer.append(experience) # Append the experience - APPEND TO THE BUFFER
        else: # If the buffer is full - IF FULL
            self.buffer[self.index] = experience # Replace the experience at the current index - REPLACE THE OLDEST

        # Add experience to the SumTree - ADD TO THE TREE
        self.tree.add(priority, experience) # Add to the tree - ADD TO THE SUM TREE

        # Update index - UPDATE THE INDEX
        self.index = (self.index + 1) % self.capacity # Increment the index - CIRCULAR BUFFER

    def sample(self, batch_size):
        """Sample a batch from the replay buffer."""
        batch = [] # Initialize the batch - BATCH LIST
        idxs = [] # Initialize the indices - INDICES LIST
        weights = [] # Initialize the weights - WEIGHTS LIST

        # Sample experiences from the SumTree with proportional prioritization - SAMPLE FROM THE TREE
        for _ in range(batch_size): # Iterate batch_size times - ITERATE
            idx, priority, experience = self.tree.sample() # Get sample from the tree - SAMPLE FROM TREE
            batch.append(experience) # Append the experience - ADD TO THE BATCH
            idxs.append(idx) # Append the index - ADD TO THE INDICES
            weights.append(self._get_weight(priority)) # Append the weight - ADD TO THE WEIGHTS

        # Convert weights to a tensor - CONVERT TO A TENSOR
        weights = tf.convert_to_tensor(weights, dtype=tf.float32) # Convert to a TensorFlow tensor - CONVERT THE WEIGHTS
        return batch, idxs, weights # Return the batch, indices, and weights - RETURN THE VALUES

    def update(self, idx, priority):
        """Update the priority of an experience in the SumTree."""
        self.tree.update(idx, priority) # Update the priority - UPDATE SUM TREE

    def _get_weight(self, priority):
        """Calculate the weight for an experience based on its priority."""
        # This is a simple placeholder; you might need to tweak this based on your algorithm. - IMPLEMENT AS NEEDED
        return priority # Return the priority - RETURN THE WEIGHT

    def __len__(self):
        """Returns the number of experiences in the buffer."""
        return len(self.buffer) # Return the length of the buffer - RETURN THE SIZE

# ---------------- NoisyLinear Class ---------------- #
class NoisyLinear(layers.Layer):
    """Noisy layer for exploration."""
    def __init__(self, in_features, out_features, std_init=0.5):
        """Initialize the NoisyLinear layer."""
        super(NoisyLinear, self).__init__() # Call the parent class constructor - CALL SUPER
        self.in_features = in_features # Store the input features - INPUT SIZE
        self.out_features = out_features # Store the output features - OUTPUT SIZE
        self.std_init = std_init # Store the standard deviation - STANDARD DEVIATION

        self.weight_mu = self.add_weight(shape=(out_features, in_features), # DEFINE WEIGHT MU
                                         initializer='uniform',  # Or glorot_uniform - INITIALIZER
                                         trainable=True, # Trainable - TRAINABLE PARAMETER
                                         name='weight_mu') # Name - NAME OF THE WEIGHT
        self.weight_sigma = self.add_weight(shape=(out_features, in_features), # DEFINE WEIGHT SIGMA
                                            initializer=initializers.Constant(std_init / np.sqrt(in_features)), # Initializer - INITIALIZER
                                            trainable=True, # Trainable - TRAINABLE PARAMETER
                                            name='weight_sigma') # Name - NAME
        self.bias_mu = self.add_weight(shape=(out_features,), # DEFINE BIAS MU
                                       initializer='uniform', # Initializer - INITIALIZER
                                       trainable=True, # Trainable - TRAINABLE
                                       name='bias_mu') # Name - NAME
        self.bias_sigma = self.add_weight(shape=(out_features,), # DEFINE BIAS SIGMA
                                          initializer=initializers.Constant(std_init / np.sqrt(in_features)), # Initializer - INITIALIZER
                                          trainable=True, # Trainable - TRAINABLE
                                          name='bias_sigma') # Name - NAME

        self.weight_epsilon = tf.Variable(tf.zeros((out_features, in_features)), trainable=False, name='weight_epsilon') # DEFINE WEIGHT EPSILON
        self.bias_epsilon = tf.Variable(tf.zeros((out_features,)), trainable=False, name='bias_epsilon') # DEFINE BIAS EPSILON

    def reset_noise(self):
        """Reset the noise."""
        self.weight_epsilon.assign(tf.random.normal((self.out_features, self.in_features))) # Assign new noise to weight epsilon - RANDOM WEIGHT NOISE
        self.bias_epsilon.assign(tf.random.normal((self.out_features,))) # Assign new noise to bias epsilon - RANDOM BIAS NOISE

    def call(self, x):
        """Forward pass."""
        self.reset_noise()  # Generate new noise each forward pass - GENERATE NEW NOISE
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon # Calculate the weight - WEIGHT WITH NOISE
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon # Calculate the bias - BIAS WITH NOISE
        return tf.matmul(x, tf.transpose(weight)) + bias  # tf.matmul and transpose for correct matrix multiplication - MATRIX MULTIPLICATION

# ---------------- CNN_DQN Class ---------------- #
class CNN_DQN(tf.keras.Model):
    """CNN-based DQN model."""
    def __init__(self, input_shape, num_actions):
        """Initialize the CNN_DQN model."""
        super(CNN_DQN, self).__init__() # Call the parent class constructor - CALL SUPER
        self.input_shape = input_shape  # (H, W, C) - INPUT SHAPE
        self.num_actions = num_actions # Number of possible actions - NUMBER OF ACTIONS

        self.conv_layers = models.Sequential([ # DEFINE THE CONVOLUTIONAL LAYERS
            layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=self.input_shape), # First conv layer - CONVOLUTIONAL LAYER
            layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'), # Second conv layer - CONVOLUTIONAL LAYER
            layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu') # Third conv layer - CONVOLUTIONAL LAYER
        ])

        # Compute dynamically - CALCULATE DYNAMICALLY
        self.fc_input_size = self._get_conv_out(input_shape) # Get the flattened size - FLATTENED SIZE

        self.fc_layers = models.Sequential([ # DEFINE THE FULLY CONNECTED LAYERS
            NoisyLinear(self.fc_input_size, 512), # First fully connected layer - FULLY CONNECTED LAYER
            layers.ReLU(), # ReLU activation - RELU ACTIVATION
            NoisyLinear(512, num_actions) # Second fully connected layer - OUTPUT LAYER
        ])

    def _get_conv_out(self, shape):
        """Compute the size of the flattened convolutional output."""
        # Create a dummy input tensor - DUMMY INPUT
        dummy_input = tf.zeros((1,) + shape)  # (batch, height, width, channels) - CREATE DUMMY INPUT
        # Pass the dummy input through the convolutional layers - PASS THROUGH CONV LAYERS
        conv_out = self.conv_layers(dummy_input) # Pass the dummy input - GET THE OUTPUT
        # Flatten the output and return the size - FLATTEN THE OUTPUT
        return np.prod(conv_out.shape[1:]) # Get the product of the shape - CALCULATE THE SIZE

    def call(self, x):
        """Forward pass."""
        x = self.conv_layers(x) # Convolutional layers - CONVOLUTIONAL LAYERS
        x = tf.reshape(x, (tf.shape(x)[0], -1))  # Flatten the output - FLATTEN THE OUTPUT
        return self.fc_layers(x) # Fully connected layers - FULLY CONNECTED LAYERS

# ---------------- CNN_DQN_Agent_TF Class ---------------- #
class CNN_DQN_Agent_TF:
    """CNN-DQN agent with Prioritized Experience Replay."""
    def __init__(self, input_shape, num_actions, lr=0.0001, gamma=0.99, batch_size=32, model_path=None, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """Initialize the CNN_DQN_Agent_TF."""
        self.input_shape = input_shape # Store the input shape - INPUT SHAPE
        self.num_actions = num_actions # Store the number of actions - NUMBER OF ACTIONS
        self.gamma = gamma # Store the discount factor - DISCOUNT FACTOR
        self.batch_size = batch_size # Store the batch size - BATCH SIZE

        # Model and target model for DQN - DEFINE THE MODELS
        self.model = CNN_DQN(input_shape, num_actions) # CREATE THE MAIN MODEL
        self.target_model = CNN_DQN(input_shape, num_actions) # CREATE THE TARGET MODEL
        self.optimizer = optimizers.Adam(learning_rate=lr) # DEFINE THE OPTIMIZER
        self.replay_buffer = PER_ReplayBuffer(100000) # DEFINE THE REPLAY BUFFER

        self.model_path = model_path # DEFINE THE MODEL PATH
        if model_path and os.path.exists(model_path + ".keras"): # Check if the model exists - CHECK IF EXISTS
            self.load(model_path) # Load the model - LOAD THE MODEL

        # Epsilon for epsilon-greedy policy - EPSILON GREEDY
        self.epsilon = epsilon_start # Initial epsilon value - INITIAL VALUE
        self.epsilon_end = epsilon_end # Final epsilon value - FINAL VALUE
        self.epsilon_decay = epsilon_decay # Epsilon decay rate - DECAY RATE

        # Copy weights from the model to the target model - COPY WEIGHTS
        self.update_target_model() # Update the target model - UPDATE THE TARGET MODEL

    def get_action(self, state):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon: # Check if we should explore - EXPLORE
            return np.random.randint(self.num_actions)  # Random action (exploration) - RANDOM ACTION

        # Add batch dimension to state and convert to TensorFlow tensor - CONVERT TO TENSOR
        state = np.expand_dims(state, axis=0).astype(np.float32)  # Add batch dimension - ADD DIMENSION
        state = tf.convert_to_tensor(state) # Convert to TensorFlow tensor - CONVERT TO TENSOR

        q_values = self.model(state)  # Forward pass to get Q-values - GET Q-VALUES
        return np.argmax(q_values.numpy()).item()  # Convert to NumPy and get action - CHOOSE ACTION

    def update_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) # Decay epsilon - DECAY EPSILON

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the PER buffer."""
        error = abs(reward)  # Placeholder error - PLACEHOLDER ERROR
        self.replay_buffer.add(error, (state, action, reward, next_state, done)) # Add to the replay buffer - ADD TO BUFFER

    def learn(self):
        """Update the model using experiences sampled from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size: # Check if there are enough samples - CHECK SIZE
            return # Return if not enough samples - NEED MORE SAMPLES

        # Sample a batch of experiences - SAMPLE A BATCH
        batch, idxs, weights = self.replay_buffer.sample(self.batch_size) # Sample the replay buffer - SAMPLE FROM BUFFER

        # Convert lists to NumPy arrays and then to TensorFlow tensors - CONVERT TO NUMPY ARRAYS
        states = np.array([b[0] for b in batch]).astype(np.float32)  # Ensure float32 dtype - CONVERT TO FLOAT32
        actions = np.array([b[1] for b in batch]).astype(np.int32)  # Ensure int32 dtype - CONVERT TO INT32
        rewards = np.array([b[2] for b in batch]).astype(np.float32) # CONVERT TO FLOAT32
        next_states = np.array([b[3] for b in batch]).astype(np.float32)  # Ensure float32 dtype - CONVERT TO FLOAT32
        dones = np.array([b[4] for b in batch]).astype(np.float32) # CONVERT TO FLOAT32

        states = tf.convert_to_tensor(states) # CONVERT TO TENSOR
        actions = tf.convert_to_tensor(actions) # CONVERT TO TENSOR
        rewards = tf.convert_to_tensor(rewards) # CONVERT TO TENSOR
        next_states = tf.convert_to_tensor(next_states) # CONVERT TO TENSOR
        dones = tf.convert_to_tensor(dones) # CONVERT TO TENSOR

        # Open a GradientTape. - OPEN GRADIENT TAPE
        with tf.GradientTape() as tape: # DEFINE GRADIENT TAPE
            # Forward pass - FORWARD PASS
            q_values = self.model(states) # Get the Q-values - GET Q-VALUES
            q_value = tf.gather_nd(q_values, tf.reshape(tf.stack([tf.range(self.batch_size), actions], axis=1), (-1, 2)))  # Indexing with tf.gather_nd - GATHER Q-VALUES
            #next_q_values = self.target_model(next_states)
            #next_q_value = tf.reduce_max(next_q_values, axis=1)  # Maximum Q-value for each next state

            # Calculate next q values using the target model. - CALCULATE NEXT Q-VALUES
            next_q_values = self.target_model(next_states) # Get the next Q-values - GET NEXT Q-VALUES
            next_actions = tf.argmax(next_q_values, axis=1)  # This will select the action that maximizes the Q value in the next state - GET NEXT ACTIONS
            next_q_value = tf.gather_nd(next_q_values, tf.reshape(tf.stack([tf.range(self.batch_size), next_actions], axis=1), (-1, 2))) #tf.reduce_max(next_q_values, axis=1) - GET NEXT Q-VALUE

            # Compute target: reward + gamma * max(Q_next) * (1 - done) - CALCULATE THE TARGET
            target = rewards + self.gamma * next_q_value * (1 - dones) # Calculate the target - TARGET VALUE

            # Calculate the loss (TD error) - CALCULATE THE LOSS
            td_error = target - q_value # Calculate the TD error - TD ERROR
            loss = tf.reduce_mean(weights * tf.square(td_error))  # Weighted MSE loss - CALCULATE THE LOSS

        # Get gradients of loss wrt the model's trainable variables - GET THE GRADIENTS
        grads = tape.gradient(loss, self.model.trainable_variables) # Get the gradients - GRADIENTS

        # Apply gradients to update the model's variables - APPLY GRADIENTS
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # Apply the gradients - APPLY GRADIENTS

        # Update priorities in the buffer - UPDATE PRIORITIES
        for i, idx in enumerate(idxs): # Iterate over the indices - ITERATE
            self.replay_buffer.update(idx, abs(td_error[i].numpy())) # Update the priority - UPDATE THE REPLAY BUFFER

        # Update epsilon - UPDATE EPSILON
        self.update_epsilon() # Update epsilon - DECAY EPSILON

    def update_target_model(self):
        """Update the target model with the current model weights."""
        self.target_model.set_weights(self.model.get_weights()) # Update the target model - COPY THE WEIGHTS

    def load(self, model_path):
        """Load the model from the specified path."""
        try: # TRY LOADING THE MODEL
            self.model = models.load_model(model_path + ".keras") # Load the model - LOAD THE WEIGHTS
            self.target_model = models.load_model(model_path + ".keras") # LOAD THE MODEL
        except Exception as e: # HANDLE EXCEPTIONS
            print(f"Error loading model: {e}") # Print the error message - PRINT MESSAGE

    def save(self, model_path):
        """Save the model to the specified path."""
        self.model.save(model_path + ".keras") # Save the model - SAVE THE WEIGHTS

# # Example Usage (Adapt this to your environment) #COMMENTED OUT
# input_shape = (84, 84, 4)  # Example input shape (height, width, channels)
# num_actions = 4  # Example number of actions
# agent = CNN_DQN_Agent(input_shape, num_actions, model_path="cnn_dqn_model.keras")

# # Sample usage (replace with your environment interaction loop)
# state = np.random.rand(*input_shape).astype(np.float32)
# for i in range(100):
#     action = agent.get_action(state)
#     next_state = np.random.rand(*input_shape).astype(np.float32)
#     reward = np.random.randn()
#     done = np.random.choice([True, False])
#     agent.remember(state, action, reward, next_state, done)
#     agent.learn()
#     state = next_state

#     if i % 10 == 0:
#         agent.update_target_model()
#         print(f"Saving model at iteration {i}")
#         agent.save("cnn_dqn_model.keras")
