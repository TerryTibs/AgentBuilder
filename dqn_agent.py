import tensorflow as tf  # Import TensorFlow library - FOR DEEP LEARNING
import numpy as np  # Import NumPy library - FOR NUMERICAL OPERATIONS
import random  # Import random library - FOR RANDOM NUMBER GENERATION
from collections import deque  # Use deque for efficient memory handling - FOR EFFICIENT QUEUE OPERATIONS

# ---------------- DQN Agent Class ---------------- #
class DQNAgent:
    """
    Implements a Deep Q-Network (DQN) agent for reinforcement learning.
    """

    def __init__(self, state_size, action_size, learning_rate=0.001, model_path='model.h5'):
        """
        Initializes the Deep Q-Network (DQN) agent.

        Parameters:
        - state_size: Number of features in the state representation.
        - action_size: Number of possible actions the agent can take.
        - learning_rate: Learning rate for the neural network.
        - model_path: File path to save/load the trained model.
        """
        self.state_size = state_size  # Define the state size - DIMENSIONALITY OF THE STATE SPACE
        self.action_size = action_size  # Define the action size - NUMBER OF POSSIBLE ACTIONS
        self.learning_rate = learning_rate  # Define the learning rate - HOW QUICKLY THE NETWORK LEARNS
        self.gamma = 0.99  # Discount factor for future rewards - HOW MUCH TO VALUE FUTURE REWARDS
        self.epsilon = 1.0  # Initial exploration rate - START FULLY EXPLORING - HOW OFTEN TO EXPLORE INITIALLY
        self.epsilon_min = 0.001  # Minimum exploration rate - CANT STOP EXPLORING COMPLETELY - ENSURE SOME EXPLORATION CONTINUES
        self.epsilon_decay = 0.9995  # Decay rate for exploration-exploitation trade off - HOW QUICKLY TO STOP EXPLORING - CONTROLS EXPLORATION OVER TIME
        self.batch_size = 500  # Mini-batch size for training - HOW MANY SAMPLES TO LEARN FROM - SIZE OF THE TRAINING BATCH
        self.memory_size = 10000  # Maximum size of the replay buffer - HOW MUCH EXPERIENCE TO REMEMBER - CAPACITY OF THE REPLAY MEMORY
        self.replay_buffer = deque(maxlen=self.memory_size)  # Experience replay memory - WHERE WE STORE EXPERIENCE - STORES EXPERIENCES FOR TRAINING
        self.model_path = model_path  # Define the model path - WHERE TO SAVE AND LOAD THE MODEL
        self.target_update_interval = 100 # update every 100 steps
        self.train_step = 0

        # Create the primary and target networks - BUILD THE NEURAL NETWORKS
        self.model = self._build_model()  # Build the primary model - THE MAIN Q-NETWORK
        self.target_model = self._build_model()  # Build the target model - USED FOR STABLE TARGETS
        self.update_target_model()  # Copy weights to target model - INITIALIZE THEM EQUALLY - SYNCHRONIZE TARGET NETWORK

    def _build_model(self):
        """
        Builds a deep neural network for Q-learning.

        Returns:
        - Compiled TensorFlow Keras model.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),  # Input layer and first hidden layer - INPUT LAYER WITH RELU ACTIVATION
            tf.keras.layers.Dense(64, activation='relu'),  # Second hidden layer
            # tf.keras.layers.Dense(64, activation='relu'),  # Third hidden layer
            # tf.keras.layers.Dense(64, activation='relu'),  # Fourth hidden layer
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Output layer for Q-values - ONE VALUE PER ACTION
        ])

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))  # COMPILE IT! - CONFIGURE THE TRAINING PROCESS
        return model  # Return the model - RETURN THE BUILT MODEL

    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience in replay buffer for training.

        Parameters:
        - state: Current state of the environment.
        - action: Action taken.
        - reward: Reward received after taking the action.
        - next_state: New state after taking the action.
        - done: Boolean indicating if the episode ended.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))  # Append the experience to the replay buffer - STORE THE EXPERIENCE IN MEMORY

    def get_action(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        Parameters:
        - state: Current state of the environment.

        Returns:
        - Action index to be taken.
        """
        if np.random.rand() <= self.epsilon:  # CHECK IF WE SHOULD EXPLORE - RANDOM CHANCE FOR EXPLORATION
            return random.randrange(self.action_size)  # Explore (random action) - ACT RANDOMLY - CHOOSE A RANDOM ACTION
        # IF NOT EXPLORING, EXPLOIT THE LEARNED POLICY
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)  # Exploit (best action) - PREDICT BEST ACTION - GET Q-VALUES FROM THE NETWORK
        return np.argmax(q_values[0])  # Return the action with the highest Q-value - CHOOSE THE BEST ACTION

    def replay(self):
        """
        Trains the neural network using a batch of experiences from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:  # CHECK IF THERE IS ENOUGH DATA TO TRAIN - NEED ENOUGH SAMPLES IN MEMORY
            return  # Not enough data to train - NEED MORE DATA FIRST! - WAIT UNTIL WE HAVE ENOUGH DATA

        mini_batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), self.batch_size))  # Sample a mini-batch from the replay buffer - GET A RANDOM BATCH OF EXPERIENCES
        states, targets = [], []  # Initialize lists for states and targets - PREPARE DATA FOR TRAINING

        for state, action, reward, next_state, done in mini_batch:  # ITERATE OVER THE SAMPLES IN THE BATCH
            # Compute target Q-value using the target network - CALCULATE THE TARGET Q-VALUE (BASED ON BELLMAN EQUATION)
            target = reward if done else reward + self.gamma * np.amax(
                self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])  # Calculate the target Q-value - Q-LEARNING UPDATE RULE
            target_f = self.model.predict(np.expand_dims(state, axis=0), verbose=0)  # Get the current Q-values - GET THE CURRENT ESTIMATES
            target_f[0][action] = target  # Update the Q-value for the taken action - SET THE TARGET FOR THE CHOSEN ACTION

            states.append(state)  # Append the state to the list - STORE THE STATE
            targets.append(target_f[0])  # Append the target to the list - STORE THE TARGET

        # Train the model using a single epoch - PERFORM ONE TRAINING ITERATION
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)  # Train the model - UPDATE THE NETWORK WEIGHTS

        # Reduce exploration rate over time - DECAY THE EXPLORATION RATE
        if self.epsilon > self.epsilon_min:  # CHECK IF WE SHOULD STILL EXPLORE - ENSURE EXPLORATION DOESN'T GO TOO LOW
            self.epsilon *= self.epsilon_decay  # Decay epsilon - REDUCE EXPLORATION RATE

    def update_target_model(self):
        """
        Copies the weights from the primary model to the target model.
        """
        self.target_model.set_weights(self.model.get_weights())  # Update the target model weights - SYNCHRONIZE THE TARGET NETWORK

    def load(self, name):
        """
        Loads a saved model from file.

        Parameters:
        - name: Path to the saved model file.
        """
        self.model = tf.keras.models.load_model(name)  # Load the model - LOAD THE TRAINED WEIGHTS
        self.update_target_model()

    def save(self, name):
        """
        Saves the current model to file.

        Parameters:
        - name: Path to save the model.
        """
        self.model.save(name)  # Save the model - SAVE THE TRAINED WEIGHTS

    def train_step_funtion(self):
        self.train_step += 1
        # Reduce exploration rate over time - DECAY THE EXPLORATION RATE
        if self.epsilon > self.epsilon_min:  # CHECK IF WE SHOULD STILL EXPLORE - ENSURE EXPLORATION DOESN'T GO TOO LOW
            self.epsilon *= self.epsilon_decay  # Decay epsilon - REDUCE EXPLORATION RATE
        if self.train_step % self.target_update_interval == 0:
            self.update_target_model()
