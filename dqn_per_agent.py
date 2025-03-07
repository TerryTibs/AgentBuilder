import random  # Import the random module - FOR RANDOM NUMBER GENERATION
import numpy as np  # Import the NumPy module - FOR NUMERICAL OPERATIONS
import tensorflow as tf  # Import TensorFlow library - FOR DEEP LEARNING
from tensorflow.keras.models import Sequential  # Import Sequential from TensorFlow Keras - FOR CREATING SEQUENTIAL MODELS
from tensorflow.keras.layers import Dense, Flatten  # Import Dense and Flatten layers - FOR BUILDING THE NEURAL NETWORK
from tensorflow.keras.optimizers import Adam  # Import the Adam optimizer - FOR TRAINING THE NETWORK
from collections import deque  # Import deque from collections - FOR EFFICIENT QUEUE OPERATIONS

# ---------------- DQN Agent Class ---------------- #
class DQN_PER_Agent:
    """
    Deep Q-Network (DQN) agent with Prioritized Experience Replay (PER)
    """

    def __init__(self, state_size, action_size, model_path=None):
        """
        Initializes the DQN agent with PER.

        Parameters:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            model_path (str, optional): Path to load/save the model. Defaults to None.
        """
        self.state_size = state_size  # Number of state variables - DIMENSION OF THE STATE SPACE
        self.action_size = action_size  # Number of possible actions - NUMBER OF ACTIONS
        self.gamma = 0.99  # Discount factor - DISCOUNT FACTOR FOR FUTURE REWARDS
        self.epsilon = 1.0  # Exploration rate - INITIAL EXPLORATION RATE
        self.epsilon_min = 0.01  # Minimum exploration rate - MINIMUM EXPLORATION RATE
        self.epsilon_decay = 0.999  # Decay rate for exploration - EXPLORATION DECAY RATE
        self.learning_rate = 0.001  # Learning rate - LEARNING RATE
        self.batch_size = 100000  # Training batch size - TRAINING BATCH SIZE
        self.memory_size = 1000000  # Replay buffer size - REPLAY BUFFER SIZE
        self.replay_buffer = deque(maxlen=self.memory_size)  # Experience replay memory - REPLAY MEMORY
        self.alpha = 0.6  # Prioritization exponent - PRIORITIZATION EXPONENT
        self.beta = 0.4  # Importance sampling exponent - IMPORTANCE SAMPLING EXPONENT
        self.beta_increment = 0.001  # Beta increment per step - BETA INCREMENT PER STEP
        self.priorities = deque(maxlen=self.memory_size)  # Stores priority values - PRIORITIES
        self.target_update_interval = 100  # Update target network every N steps
        self.train_step = 0  # Counter for training steps

        # Initialize the neural network model - BUILD THE MODEL
        self.model = self._build_model()  # Build the main model - BUILD THE MAIN MODEL
        self.target_model = self._build_model()  # Build the target model - BUILD THE TARGET MODEL
        self.update_target_model()  # Update the target model - UPDATE THE TARGET MODEL

        if model_path:  # Check if a model path is provided - CHECK FOR MODEL PATH
            self.load(model_path)  # Load existing model if available - LOAD THE MODEL

    def _build_model(self):
        """Builds a simple neural network for DQN."""
        model = Sequential([
            Flatten(input_shape=(self.state_size,)),  # Flatten the input - FLATTEN THE INPUT
            Dense(64, activation='relu'),  # First dense layer - FIRST DENSE LAYER
            Dense(64, activation='relu'),  # Second dense layer - SECOND DENSE LAYER
            Dense(64, activation='relu'),  # Third dense layer - THIRD DENSE LAYER
            Dense(64, activation='relu'),  # Forth dense layer - FORTH DENSE LAYER
            Dense(self.action_size, activation='linear')  # Output layer - OUTPUT LAYER
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Compile the model - COMPILE THE MODEL
        return model  # Return the model - RETURN THE MODEL

    def update_target_model(self):
        """Copies the weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())  # Set the weights of the target model - COPY THE WEIGHTS

    def get_action(self, state):
        """Returns an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:  # Explore if a random number is less than epsilon - EXPLORE
            return random.randrange(self.action_size)  # Return a random action - RANDOM ACTION
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)  # Predict the Q-values - PREDICT Q-VALUES
        return np.argmax(q_values[0])  # Return the action with the highest Q-value - BEST ACTION

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in replay memory with priority."""
        priority = max(self.priorities, default=1.0)  # Use max priority if memory is empty - INITIAL PRIORITY
        self.replay_buffer.append((state, action, reward, next_state, done))  # Append the experience to the replay buffer - STORE THE EXPERIENCE
        self.priorities.append(priority)  # Append the priority - STORE THE PRIORITY

    def replay(self):
        """Trains the network using Prioritized Experience Replay."""
        if len(self.replay_buffer) < self.batch_size:  # Check if there are enough samples in the replay buffer - CHECK FOR ENOUGH DATA
            return  # Return if there are not enough samples - NOT ENOUGH DATA

        priorities = np.array(self.priorities)  # Convert the priorities to a NumPy array - CONVERT TO NUMPY ARRAY
        probs = priorities ** self.alpha  # Calculate the probabilities - CALCULATE PROBABILITIES
        probs /= probs.sum()  # Normalize the probabilities - NORMALIZE PROBABILITIES

        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)  # Choose the indices - CHOOSE INDICES
        batch = [self.replay_buffer[i] for i in indices]  # Create the batch - CREATE THE BATCH

        states, actions, rewards, next_states, dones = zip(*batch)  # Unpack the batch - UNPACK THE BATCH
        states = np.array(states)  # Convert the states to a NumPy array - CONVERT TO NUMPY ARRAY
        next_states = np.array(next_states)  # Convert the next states to a NumPy array - CONVERT TO NUMPY ARRAY

        q_values = self.model.predict(states, verbose=0)  # Predict the target Q-values - PREDICT Q-VALUES
        next_q_values = self.target_model.predict(next_states, verbose=0)  # Predict the next Q-values - PREDICT NEXT Q-VALUES

        # importance sampling weights
        weights = (len(self.replay_buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize for stability

        for i, index in enumerate(indices):  # Iterate over the indices - ITERATE
            if dones[i]:  # If the episode is done - EPISODE IS DONE
                q_values[i][actions[i]] = rewards[i]  # Set the target Q-value to the reward - SET TARGET Q-VALUE
            else:  # If the episode is not done - EPISODE IS NOT DONE
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])  # Calculate the target Q-value - CALCULATE TARGET Q-VALUE

            # Update priority - UPDATE PRIORITY
            self.priorities[index] = abs(
                rewards[i] + self.gamma * np.max(next_q_values[i]) - q_values[i][actions[i]]) + 1e-5  # Update the priority - UPDATE PRIORITY

        # Train the model with importance sampling weights
        with tf.GradientTape() as tape:
            q_preds = self.model(states)
            loss = tf.keras.losses.MeanSquaredError()(q_values, q_preds)
            loss = tf.reduce_mean(weights * loss)  # Apply importance sampling weights

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update epsilon and beta
        if self.epsilon > self.epsilon_min:  # Check if epsilon is greater than the minimum - CHECK EPSILON
            self.epsilon *= self.epsilon_decay  # Decay epsilon - DECAY EPSILON

        if self.beta < 1.0:
            self.beta += self.beta_increment

        # Update target model periodically
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.update_target_model()

    def save(self, model_path):
        """Saves the trained model."""
        self.model.save(model_path)  # Save the model - SAVE THE MODEL
        print(f"Model saved at {model_path}")  # Print a message - PRINT MESSAGE

    def load(self, model_path):
        """Loads a pre-trained model if available."""
        try:  # TRY LOADING THE MODEL
            self.model = tf.keras.models.load_model(model_path)  # Load the model - LOAD THE MODEL
            self.update_target_model()  # Update the target model - UPDATE TARGET MODEL
            print(f"Model loaded from {model_path}")  # Print a message - PRINT MESSAGE
        except Exception as e:  # HANDLE EXCEPTIONS
            print(f"Error loading model: {e}")  # Print an error message - PRINT MESSAGE
