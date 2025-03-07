# Import required libraries
import pygame  # For game creation
import random  # For random number generation
import numpy as np  # For numerical operations
from abstract_game import AbstractGame  # Import the abstract game class

# ---------------- Constants ---------------- #
GRID_WIDTH = 800  # Define grid width
GRID_HEIGHT = 600  # Define grid height
GRID_SIZE = 10  # Define grid size
GRID_COLOR = (0, 0, 0)  # Black color
SNAKE_COLOR = (0, 255, 0)  # Green color
APPLE_COLOR = (255, 0, 0)  # Red color
SCORE_COLOR = (255, 255, 255)  # White color
FONT_SIZE = 14  # Font size for score
FONT_NAME = 'Arial'  # Font name for score
DEFAULT_FPS = 60  # Default frames per second
INITIAL_SNAKE_LENGTH = 1  # Initial snake length
GRID_COLS = GRID_WIDTH // GRID_SIZE  # Calculate number of columns
GRID_ROWS = GRID_HEIGHT // GRID_SIZE  # Calculate number of rows
STATE_SIZE = 12  # Define the state size
ACTION_SIZE = 4  # Define the action size
SNAKE_SPEED = 5  # Snake moves once every 'snake_speed' frames
BASE_LIMIT = 20000  # The amount the frame iteration counts by
NUM_APPLES = 100  # Number of apples to spawn
DEATH_PENALTY = -10000  # Penalty for dying
APPLE_REWARD = 2000  # Reward for eating an apple
WALL_PROXIMITY_PENALTY_SCALE = 0.001  # Scale for wall proximity penalty
APPLE_PROXIMITY_REWARD_SCALE = 0.001  # Scale for apple proximity reward


# ---------------- Snake Game Class ---------------- #
class SnakeGame(AbstractGame):
    """
    Implements the Snake game, inheriting from AbstractGame.
    """

    def __init__(self):
        """
        Initializes the Snake game, setting up Pygame and game variables.
        """
        pygame.init()  # Initialize pygame
        try:
            self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))  # Create the game window
            pygame.display.set_caption("Snake Game - AI Learning")  # Set the window title
            self.clock = pygame.time.Clock()  # Create a clock object for controlling FPS
            self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)  # Set the font

            self.state_size = STATE_SIZE  # Define the state size
            self.action_size = ACTION_SIZE  # Define the action size

            # Independent speed settings
            self.fps = DEFAULT_FPS  # Set the frames per second
            self.snake_speed = SNAKE_SPEED  # Snake moves once every 'snake_speed' frames
            self.move_counter = 0  # Counter to track movement intervals

            self.reset()  # Reset the game state
            self.draw()  # Draw the initial game state
        except Exception as e:
            print(f"Error during Pygame initialization: {e}")  # Print the error message
            raise  # Raise the exception

    def get_screen_size(self):
        """
        Returns the screen size as a tuple (width, height, channels).
        """
        return (GRID_WIDTH, GRID_HEIGHT, 3)  # (width, height, RGB channels)

    def get_screen(self):
        """
        Captures the game screen as a NumPy array for CNN input.
        """
        screen_array = pygame.surfarray.array3d(pygame.display.get_surface())  # Capture screen as numpy array
        return np.transpose(screen_array, (1, 0, 2))  # Convert Pygame format to standard (H, W, C)

    def reset(self):
        """
        Resets the game state, placing the snake and apples in their initial positions.
        """
        self.snake = [(GRID_COLS // 2, GRID_ROWS // 2)]  # Initialize the snake at the center
        for i in range(1, INITIAL_SNAKE_LENGTH):
            self.snake.append((self.snake[0][0] - i, self.snake[0][1]))  # Add segments to the snake

        self.direction = (1, 0)  # Set the initial direction
        self.apples = []  # Initialize the apples list
        self.generate_apples()  # Generate the initial apples
        self.score = 0  # Reset the score
        self.apples_eaten = 0  # Reset apples eaten count
        self.game_over = False  # Reset game over flag
        self.fibonacci_rewards = [10, 10]  # Initialize fibonacci rewards (kept from original script)
        self.move_counter = 0  # Reset movement counter
        self.base_limit = BASE_LIMIT  # Base frame iteration limit - NEW: base for adaptive limit
        self.frame_iteration = 0  # Reset frame iteration - NEW: Counts frames in episode
        self.epsilon = 1.0  # Initial exploration rate

    def generate_apples(self):
        """
        Generates a list of apple positions to have NUM_APPLES apples on the grid.
        """
        self.apples = []  # clear previous apples
        while len(self.apples) < NUM_APPLES:  # Ensure there are NUM_APPLES apples
            apple_x = random.randint(0, GRID_COLS - 1)  # Generate random x coordinate
            apple_y = random.randint(0, GRID_ROWS - 1)  # Generate random y coordinate
            apple_position = (apple_x, apple_y)  # Create apple position tuple

            if apple_position not in self.snake and apple_position not in self.apples:  # Make sure apple is not on snake
                self.apples.append(apple_position)  # Add the position to the list

    def generate_single_apple(self):
        """
        Generates a single apple at a random location, avoiding the snake.
        """
        while True:
            apple_x = random.randint(0, GRID_COLS - 1)  # Generate random x coordinate
            apple_y = random.randint(0, GRID_ROWS - 1)  # Generate random y coordinate
            apple_position = (apple_x, apple_y)  # Create apple position tuple

            if apple_position not in self.snake and apple_position not in self.apples:  # Make sure apple is not on snake
                return apple_position  # Return the apple position

    def get_state(self):
        """
        Returns the current game state as a NumPy array.
        The state includes relative coordinates to the apple, distances to the walls,
        and snake proximity awareness.
        """
        head_x, head_y = self.snake[0]  # Get head coordinates
        apple_x, apple_y = self.apples[0]  # Get apple coordinates

        distance_to_left = head_x / GRID_COLS  # Distance to the left wall
        distance_to_right = (GRID_COLS - 1 - head_x) / GRID_COLS  # Distance to the right wall
        distance_to_up = head_y / GRID_ROWS  # Distance to the top wall
        distance_to_down = (GRID_ROWS - 1 - head_y) / GRID_ROWS  # Distance to the bottom wall

        rel_x = (apple_x - head_x) / GRID_COLS  # Relative x coordinate of apple
        rel_y = (apple_y - head_y) / GRID_ROWS  # Relative y coordinate of apple

        # Check immediate proximity to the head:
        proximity_up = 1 if (head_x, head_y - 1) in self.snake[1:] or head_y == 0 else 0  # Is there danger up?
        proximity_down = 1 if (head_x, head_y + 1) in self.snake[1:] or head_y == GRID_ROWS - 1 else 0  # Is there danger down?
        proximity_left = 1 if (head_x - 1, head_y) in self.snake[1:] or head_x == 0 else 0  # Is there danger left?
        proximity_right = 1 if (head_x + 1, head_y) in self.snake[1:] or head_x == GRID_COLS - 1 else 0  # Is there danger right?

        wall_proximity = self._distance_to_walls() / (GRID_WIDTH if GRID_WIDTH < GRID_HEIGHT else GRID_HEIGHT)  # Normalize
        apple_proximity = self._distance_to_closest_apple() / np.sqrt(GRID_WIDTH ** 2 + GRID_HEIGHT ** 2)  # Normalize

        return np.array([rel_x, rel_y, distance_to_left, distance_to_right, distance_to_up, distance_to_down,
                         proximity_up, proximity_down, proximity_left, proximity_right, wall_proximity,
                         apple_proximity])  # Return the state as a numpy array

    def step(self, action):
        """
        Updates the game state based on the given action.
        Snake only moves if enough frames have passed.
        """
        self.clock.tick(self.fps)  # Maintain constant FPS
        self.move_counter += 1  # Increment movement counter

        # Move the snake only if enough frames have passed
        if self.move_counter < self.snake_speed:
            return -0.01, self.game_over  # Minor penalty for waiting

        self.move_counter = 0  # Reset counter

        # Convert action index to movement direction
        if action == 0:
            new_direction = (0, -1)  # Up
        elif action == 1:
            new_direction = (0, 1)  # Down
        elif action == 2:
            new_direction = (-1, 0)  # Left
        elif action == 3:
            new_direction = (1, 0)  # Right
        else:
            new_direction = self.direction  # No change

        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction  # Update direction

        head_x, head_y = self.snake[0]  # Get current head coordinates
        new_head = (head_x + self.direction[0], head_y + self.direction[1])  # Calculate new head coordinates

        if new_head[0] < 0 or new_head[0] >= GRID_COLS or new_head[1] < 0 or new_head[1] >= GRID_ROWS:
            self.game_over = True  # Game over if snake hits the wall
            return DEATH_PENALTY, self.game_over  # HUGE PENALTY FOR DYING

        if new_head in self.snake[1:]:
            self.game_over = True  # Game over if snake hits itself
            return DEATH_PENALTY, self.game_over  # HUGE PENALTY FOR DYING

        self.snake.insert(0, new_head)  # Insert the new head into the snake

        ate_apple = False  # Flag to check if an apple was eaten
        if new_head in self.apples:
            self.apples.remove(new_head)  # Remove the apple
            self.apples.append(self.generate_single_apple())  # Add a new apple
            self.apples_eaten += 1  # Increment apples eaten count
            self.score += 1  # Increment score
            reward = self.fibonacci_rewards[-1] + self.fibonacci_rewards[-2]  # Calculate reward using fibonacci sequence
            self.fibonacci_rewards.append(reward)  # Append to fibonacci rewards
            reward += APPLE_REWARD  # add apple reward
            ate_apple = True  # Set the flag

        if not ate_apple:
            self.snake.pop()  # Remove the last segment of the snake if no apple was eaten
            reward = -0.01  # time penalty

        # New reward structure elements:
        # Calculate distance to apple
        apple_distance = np.sqrt((new_head[0] - self.apples[0][0] * GRID_SIZE) ** 2 + (new_head[1] - self.apples[0][1] * GRID_SIZE) ** 2)

        if len(self.snake) > 1:
            old_head = self.snake[1]
            old_apple_distance = np.sqrt((old_head[0] - self.apples[0][0] * GRID_SIZE) ** 2 + (old_head[1] - self.apples[0][1] * GRID_SIZE) ** 2)

            if apple_distance < old_apple_distance:
                reward += 1  # Bonus for getting closer to the apple

            elif apple_distance > old_apple_distance:
                reward -= 1  # Penalty for getting further from the apple
        else:
            # Snake has only one segment, so no previous distance to compare
            old_apple_distance = apple_distance  # Or a large number like float('inf')

        # Proximity-based rewards/penalties
        wall_distance = self._distance_to_walls()
        apple_distance = self._distance_to_closest_apple()

        # Wall proximity penalty (adjust the scale as needed)
        reward -= WALL_PROXIMITY_PENALTY_SCALE * (1 / (wall_distance + 0.001))  # Further away = smaller penalty

        # Apple proximity reward (adjust the scale as needed)
        reward += APPLE_PROXIMITY_REWARD_SCALE * (1 / (apple_distance + 0.001))  # Closer = higher reward

        self.frame_iteration += 1  # Increment the frame iteration counter - NEW
        adaptive_limit = self.base_limit + 5 * len(self.snake)  # Calculate adaptive limit - NEW
        if self.frame_iteration > adaptive_limit:  # Check if exceeded adaptive limit - NEW
            self.game_over = True  # If it exceeds the limit, the game is over - NEW
            reward -= 10  # Small negative reward for timing out

        return reward, self.game_over  # Return reward

    def _distance_to_walls(self):
        """
        Calculates the minimum distance from the snake's head to any wall.
        """
        head_x, head_y = self.snake[0]
        distance_to_left = head_x
        distance_to_right = GRID_WIDTH - head_x
        distance_to_up = head_y
        distance_to_down = GRID_HEIGHT - head_y
        return min(distance_to_left, distance_to_right, distance_to_up, distance_to_down)

    def _distance_to_closest_apple(self):
        """
        Calculates the distance from the snake's head to the closest apple.
        """
        head_x, head_y = self.snake[0]
        min_distance = float('inf')  # Initialize with a very large value

        for apple_x, apple_y in self.apples:
            distance = np.sqrt((head_x - apple_x * GRID_SIZE) ** 2 + (head_y - apple_y * GRID_SIZE) ** 2)
            min_distance = min(min_distance, distance)

        return min_distance

    def draw(self):
        """
        Renders the game state visually using Pygame.
        """
        self.screen.fill(GRID_COLOR)  # Fill screen with grid color

        for x, y in self.snake:
            pygame.draw.rect(self.screen, SNAKE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Draw the snake

        for x, y in self.apples:
            pygame.draw.rect(self.screen, APPLE_COLOR, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))  # Draw the apples

        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR)  # Render the score
        self.screen.blit(score_text, (10, 10))  # Blit the score to the screen

        pygame.display.flip()  # Update the full display Surface to the screen

    def quit(self):
        """
        Quits Pygame.
        """
        pygame.quit()  # Quit pygame

    def is_done(self):
        """
        Returns True if the game is over, False otherwise.
        """
        return self.game_over  # Return game over status

    def get_action_space(self):
        """
        Returns the number of possible actions.
        """
        return self.action_size  # Return the action size

    def act(self, state):
        """
        Epsilon-greedy action selection.
        """
        if random.random() <= self.epsilon:
            # Explore: Choose a random action
            return random.randrange(self.action_size)
        else:
            # Exploit: Choose the best action (to be implemented with your DQN model)
            # This is just a placeholder; replace with your model's prediction
            return random.randrange(self.action_size)

    def update_epsilon(self, decay_rate):
        """
        Decays the exploration rate.
        """
        self.epsilon = max(0.01, self.epsilon * (1 - decay_rate))  # Ensure it doesn't go below 0.01
