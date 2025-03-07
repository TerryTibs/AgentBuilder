import pygame  # Handles game rendering and events - USED FOR VISUALIZING THE GAME
import os  # Used for file operations like saving/loading models - USED FOR SAVING AND LOADING THE TRAINED MODEL
import time  # Measures training duration - USED TO TRACK HOW LONG TRAINING TAKES
import tensorflow as tf  # Deep learning framework - THE CORE DEEP LEARNING LIBRARY
from dqn_agent import DQNAgent  # Import the standard Deep Q-Network (DQN) agent - IMPORT THE BASIC DQN AGENT
from dqn_per_agent import DQN_PER_Agent  # Import DQN agent with Prioritized Experience Replay - IMPORT THE DQN AGENT WITH PER
from cnn_dqn_agent import CNN_DQN_Agent  # Import CNN-DQN agent for raw pixel input - IMPORT THE CNN-BASED AGENT
from snake import SnakeGame  # Import the Snake game environment - IMPORT THE SNAKE GAME
from pacman import PacmanGame  # Import the Pacman game environment - IMPORT THE PACMAN GAME

# Import the new CNN-DQN agent from cnn_dqn_agent_tf.py - ATTEMPT TO IMPORT A TENSORFLOW-SPECIFIC CNN AGENT
try:
    from cnn_dqn_agent_tf import CNN_DQN_Agent_TF # TRY TO IMPORT THE AGENT
except ImportError: # IF IT FAILS
    print("Error: cnn_dqn_agent_tf.py not found or has import errors. Agent 'cnn_dqn_tf' will not be available.") # PRINT AN ERROR MESSAGE
    CNN_DQN_Agent_TF = None # SET THE AGENT TO NONE

# Ensure TensorFlow doesn't use GPU (forces CPU usage) - FORCE CPU USAGE FOR TRAINING (CAN BE REMOVED IF GPU IS DESIRED)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # DISABLE GPU

# Dictionary of available games (can add more later) - DEFINE THE AVAILABLE GAMES
GAMES = {
    "snake": SnakeGame, # SNAKE GAME
    "pacman": PacmanGame, # PACMAN GAME
}

# Dictionary of available agents - DEFINE THE AVAILABLE AGENTS
AGENTS = {
    "dqn": DQNAgent,  # Standard DQN agent - STANDARD DQN
    "dqn_per": DQN_PER_Agent,  # DQN and PER mixed - DQN WITH PRIORITIZED EXPERIENCE REPLAY
    "cnn_dqn": CNN_DQN_Agent,  # CNN-DQN agent (raw pixel input) - CNN-BASED DQN
}

# Conditionally add CNN_DQN_Agent_TF to the AGENTS dictionary - ADD THE TENSORFLOW AGENT IF IT WAS IMPORTED SUCCESSFULLY
if CNN_DQN_Agent_TF: # CHECK IF THE AGENT IS AVAILABLE
    AGENTS["cnn_dqn_tf"] = CNN_DQN_Agent_TF # ADD THE AGENT TO THE DICTIONARY

# Define training parameters - DEFINE THE TRAINING SETTINGS
EPISODES = 500000000  # Total number of training episodes - NUMBER OF TRAINING ITERATIONS
RENDER = True  # Set to False to disable game rendering (faster training) - WHETHER TO DISPLAY THE GAME DURING TRAINING
MODEL_PATH = "model.keras"  # Path where the trained model is saved - WHERE TO SAVE THE TRAINED MODEL

# ---------------- Training Script Class ---------------- #
class TrainingScript:
    """Handles training of the DQN agent on the selected game."""

    def __init__(self, game_name, agent_type):
        """
        Initializes the game and DQN agent.

        Args:
            game_name (str): The name of the game to train on.
            agent_type (str): The type of agent to use for training.
        """
        if game_name not in GAMES: # CHECK IF THE SELECTED GAME IS VALID
            raise ValueError(f"Game '{game_name}' not found. Available games: {list(GAMES.keys())}") # RAISE AN ERROR IF THE GAME IS NOT FOUND

        if agent_type not in AGENTS: # CHECK IF THE SELECTED AGENT IS VALID
            raise ValueError(f"Agent '{agent_type}' not found. Available agents: {list(AGENTS.keys())}") # RAISE AN ERROR IF THE AGENT IS NOT FOUND

        # Create an instance of the selected game - CREATE THE GAME INSTANCE
        self.game = GAMES[game_name]() # CREATE THE GAME

        # Get game state size and action space - GET THE GAME'S STATE AND ACTION SPACE
        self.state_size = self.game.state_size if agent_type != "cnn_dqn" and agent_type != "cnn_dqn_tf" else self.game.get_screen_size() # GET THE STATE SIZE DEPENDING ON THE AGENT TYPE - CNN AGENTS USE SCREEN SIZE AS STATE
        self.action_size = self.game.get_action_space() # GET THE ACTION SPACE

        # Initialize the selected agent - CREATE THE AGENT INSTANCE
        self.agent = AGENTS[agent_type](self.state_size, self.action_size, model_path=MODEL_PATH) # CREATE THE AGENT

        # Load existing model if available, otherwise create a new one - LOAD OR CREATE THE MODEL
        if os.path.exists(MODEL_PATH): # CHECK IF THE MODEL EXISTS
            print("Loading existing model...") # PRINT A MESSAGE
            self.agent.load(MODEL_PATH) # LOAD THE MODEL
        else: # IF THE MODEL DOESN'T EXIST
            print("No model detected. Creating new model...") # PRINT A MESSAGE
            self.agent.save(MODEL_PATH) # SAVE A NEW MODEL
            print("New model created and saved.") # PRINT A MESSAGE

    def train(self):
        """Runs the training loop for the agent."""
        start_time = time.time()  # Track training duration - RECORD THE START TIME

        try:
            for episode in range(EPISODES):  # Loop through each episode - ITERATE OVER THE EPISODES
                MAX_STEPS = max(1000000, episode * 1000000)  # Dynamically increase step limit - DYNAMICALLY INCREASE THE STEP LIMIT

                self.game.reset()  # Reset the game state at the beginning of each episode - RESET THE GAME
                state = self.game.get_screen() if isinstance(self.agent, CNN_DQN_Agent) or isinstance(self.agent, CNN_DQN_Agent_TF) else self.game.get_state() # GET THE INITIAL STATE - CNN AGENTS USE THE SCREEN AS INPUT
                total_reward = 0  # Track total episode reward - TRACK THE REWARD
                total_penalty = 0  # Track penalties - TRACK PENALTIES (NEGATIVE REWARDS)
                steps_taken = 0  # Track steps taken in this episode - TRACK THE NUMBER OF STEPS

                for step in range(MAX_STEPS):  # Limit the number of steps per episode - ITERATE OVER THE STEPS
                    pygame.event.pump()  # Process events to prevent game freezing - PROCESS PYGAME EVENTS

                    # Get action from the DQN agent - GET THE ACTION
                    action = self.agent.get_action(state) # GET THE ACTION FROM THE AGENT

                    # Take action in the game and receive reward + termination status - TAKE THE ACTION AND GET THE REWARD
                    reward, done = self.game.step(action) # TAKE THE ACTION

                    # Get the new state after the action - GET THE NEXT STATE
                    next_state = self.game.get_screen() if isinstance(self.agent, CNN_DQN_Agent) or isinstance(self.agent, CNN_DQN_Agent_TF) else self.game.get_state() # GET THE NEXT STATE - CNN AGENTS USE THE SCREEN

                    # Track penalties (negative rewards) - TRACK THE PENALTIES
                    if reward < 0: # CHECK IF THE REWARD IS NEGATIVE
                        total_penalty += abs(reward) # ADD THE PENALTY TO THE TOTAL

                    # Store experience in agent's memory (for training) - STORE THE EXPERIENCE
                    self.agent.remember(state, action, reward, next_state, done) # REMEMBER THE EXPERIENCE

                    # Update state for the next step - UPDATE THE STATE
                    state = next_state # UPDATE THE STATE

                    # Update total reward and step count - UPDATE THE REWARD AND STEP COUNT
                    total_reward += reward # UPDATE THE REWARD
                    steps_taken += 1 # UPDATE THE STEP COUNT

                    # Render the game if enabled - RENDER THE GAME
                    if RENDER: # CHECK IF RENDERING IS ENABLED
                        self.game.draw() # DRAW THE GAME

                    # If game is over, break out of the loop - CHECK IF THE EPISODE IS OVER
                    if done: # CHECK IF THE EPISODE IS DONE
                        print(f"Episode {episode}/{EPISODES} | Score: {total_reward} | Steps: {steps_taken} | Penalty: {total_penalty} | Epsilon: {self.agent.epsilon:.2f}") # PRINT THE TRAINING STATISTICS
                        break # BREAK OUT OF THE LOOP

                # Train the DQN agent after each episode - TRAIN THE AGENT
                train_start = time.time() # RECORD THE START TIME
                if isinstance(self.agent, CNN_DQN_Agent) or isinstance(self.agent, CNN_DQN_Agent_TF): # CHECK IF THE AGENT IS A CNN AGENT
                    self.agent.learn()  # Train CNN-DQN agent - TRAIN THE CNN AGENT
                else: # IF THE AGENT IS NOT A CNN AGENT
                    self.agent.replay()  # Train DQN or DQN_PER agent - TRAIN THE DQN AGENT
                self.agent.update_target_model()  # Update the target network - UPDATE THE TARGET NETWORK
                train_time = time.time() - train_start  # Measure training time - CALCULATE THE TRAINING TIME

                # Print training stats - PRINT THE TRAINING STATISTICS
                print(f"Training Time: {train_time:.2f}s | Memory Size: {len(self.agent.replay_buffer)}") # PRINT THE TRAINING TIME AND MEMORY SIZE
                loop_time = time.time() - start_time # CALCULATE THE LOOP TIME
                print(f"Loop {episode} lasted: {loop_time:.2f}s") # PRINT THE LOOP TIME

                # Reset start_time for next loop - RESET THE START TIME
                start_time = time.time() # RESET THE START TIME

        except KeyboardInterrupt: # HANDLE KEYBOARD INTERRUPTS
            print("Training interrupted. Saving model...") # PRINT A MESSAGE

        finally: # ALWAYS RUN THIS CODE
            # Save the trained model before exiting - SAVE THE MODEL
            print("Saving model...") # PRINT A MESSAGE
            self.agent.save(MODEL_PATH) # SAVE THE MODEL
            self.game.quit() # QUIT THE GAME
            print("Model saved. Exiting.") # PRINT A MESSAGE

if __name__ == "__main__":
    # Prompt user to choose a game - ASK THE USER TO CHOOSE A GAME
    selected_game = input(f"Choose a game to train on {list(GAMES.keys())}: ") # GET THE GAME FROM THE USER

    # Prompt user to choose an agent - ASK THE USER TO CHOOSE AN AGENT
    selected_agent = input(f"Choose an agent to use {list(AGENTS.keys())}: ") # GET THE AGENT FROM THE USER

    # Start training with the selected game and agent - START TRAINING
    trainer = TrainingScript(selected_game, selected_agent) # CREATE THE TRAINING SCRIPT
    trainer.train() # START TRAINING
