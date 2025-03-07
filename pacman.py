import pygame # Handles game rendering and events - PYGAME LIBRARY
import random # Used for random number generation - RANDOM NUMBERS
import numpy as np # Used for numerical operations - NUMPY LIBRARY
from abstract_game import AbstractGame  # Inherit from AbstractGame - INHERIT FROM ABSTRACT CLASS

# ---------------- Constants ---------------- #
# Constants for screen size - SCREEN SIZE CONSTANTS
SCREEN_WIDTH = 800 # Screen width - SCREEN WIDTH
SCREEN_HEIGHT = 600 # Screen height - SCREEN HEIGHT
GRID_COLOR = (0, 0, 0) # Background color - BACKGROUND COLOR
PACMAN_COLOR = (255, 255, 0) # Pac-Man color - PACMAN COLOR
GHOST_COLOR = (255, 0, 0) # Ghost color - GHOST COLOR
PELLET_COLOR = (255, 255, 255) # Pellet color - PELLET COLOR
WALL_COLOR = (0, 0, 255) # Wall color - WALL COLOR
SCORE_COLOR = (255, 255, 255) # Score color - SCORE COLOR
FONT_SIZE = 14 # Font size for the score - FONT SIZE
FONT_NAME = 'Arial' # Font name for the score - FONT NAME
FPS = 10 # Frames per second - FRAMES PER SECOND

# More complex maze layout - 1 = wall, 0 = empty space, 2 = pellet - MAZE LAYOUT
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # Maze row - WALLS
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1], # Maze row - EMPTY SPACE
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1], # Maze row - EMPTY SPACE
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1], # Maze row - EMPTY SPACE
    [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1], # Maze row - EMPTY SPACE
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1], # Maze row - EMPTY SPACE
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1], # Maze row - EMPTY SPACE
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1], # Maze row - EMPTY SPACE
    [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1], # Maze row - EMPTY SPACE
    [1, 0, 2, 1, 0, 0, 1, 0, 1, 2, 0, 1, 1, 0, 0, 0, 1], # Maze row - EMPTY SPACE
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1] # Maze row - WALLS
]

# ---------------- PacmanGame Class ---------------- #
class PacmanGame(AbstractGame):
    """Pac-Man game environment based on the AbstractGame class."""
    def __init__(self):
        """Initializes the Pacman game environment."""
        pygame.init() # Initializes pygame - INITIALIZE PYGAME
        
        # Adjust screen size dynamically based on the maze - ADJUST SCREEN SIZE
        self.grid_width = len(MAZE[0]) # Get width of the maze - MAZE WIDTH
        self.grid_height = len(MAZE) # Get height of the maze - MAZE HEIGHT
        self.grid_size = min(SCREEN_WIDTH // self.grid_width, SCREEN_HEIGHT // self.grid_height) # Determine grid size - GRID SIZE
        
        # Create screen based on calculated grid size - CREATE SCREEN
        self.screen = pygame.display.set_mode((self.grid_width * self.grid_size, self.grid_height * self.grid_size)) # Create the screen - CREATE THE SCREEN
        pygame.display.set_caption("Pac-Man Clone - AI Learning") # Set window caption - WINDOW CAPTION
        self.clock = pygame.time.Clock() # Create a clock object - GAME CLOCK
        self.font = pygame.font.SysFont(FONT_NAME, FONT_SIZE) # Create a font object - GAME FONT
        self.state_size = 8  # Adjusted to match the number of state variables - STATE SIZE
        self.action_size = 4  # Move up, down, left, right - ACTION SIZE
        self.reset() # Reset the game - RESET THE GAME
        self.draw() # Draw the game - DRAW THE GAME

    def reset(self):
        """Resets the game state."""
        self.pacman = (1, 1)  # Starting position for Pac-Man - PACMAN POSITION
        self.ghosts = [(7, 5), (5, 3)]  # Starting positions for ghosts - GHOST POSITIONS
        self.pellets = [(x, y) for y in range(len(MAZE)) for x in range(len(MAZE[y])) if MAZE[y][x] == 2] # Get initial positions of pellets - PELLET POSITIONS
        self.score = 0 # Set score to zero - SCORE
        self.game_over = False # Game is not over yet - GAME OVER FLAG

    def get_state(self):
        """Gets the current state of the game."""
        pac_x, pac_y = self.pacman # Get Pac-Man position - PACMAN POSITION
        ghost_positions = [ # Get ghost positions - GHOST POSITIONS
            [(gx - pac_x) / self.grid_width, (gy - pac_y) / self.grid_height] for gx, gy in self.ghosts # Calculate relative ghost positions - RELATIVE POSITIONS
        ]
        pellet_positions = [ # Get pellet positions - PELLET POSITIONS
            [(px - pac_x) / self.grid_width, (py - pac_y) / self.grid_height] for px, py in self.pellets # Calculate relative pellet positions - RELATIVE POSITIONS
        ]

        ghost_positions_flat = [coord for ghost in ghost_positions for coord in ghost] # Flatten ghost positions - FLATTEN GHOST POSITIONS
        pellet_positions_flat = [coord for pellet in pellet_positions for coord in pellet] # Flatten pellet positions - FLATTEN PELLET POSITIONS

        state = np.array(ghost_positions_flat + pellet_positions_flat + [pac_x / self.grid_width, pac_y / self.grid_height]) # Create the state array - STATE ARRAY
        return state # Return the state - RETURN THE STATE

    def step(self, action):
        """Takes a game step given the action."""
        self.clock.tick(FPS) # Limit the frame rate - FRAME RATE
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Define possible movement directions - DIRECTIONS
        move_x, move_y = directions[action] # Get the movement for the selected action - MOVEMENT
        new_pacman = (self.pacman[0] + move_x, self.pacman[1] + move_y) # Calculate the new Pac-Man position - NEW POSITION

        if 0 <= new_pacman[0] < len(MAZE[0]) and 0 <= new_pacman[1] < len(MAZE) and MAZE[new_pacman[1]][new_pacman[0]] != 1: # Check for valid move - VALID MOVE
            self.pacman = new_pacman # Update Pac-Man position - UPDATE POSITION

        if self.pacman in self.ghosts: # Check for collision with ghost - COLLISION
            self.game_over = True # Game over - GAME OVER
            return -10, self.game_over  # Penalty for being caught by a ghost - PENALTY

        if self.pacman in self.pellets: # Check if Pac-Man ate a pellet - EAT PELLET
            self.pellets.remove(self.pacman) # Remove the eaten pellet - REMOVE PELLET
            self.score += 1 # Increase the score - INCREASE SCORE
            return 1, self.game_over  # Reward for eating a pellet - REWARD

        return -0.01, self.game_over  # Small penalty to encourage faster gameplay - PENALTY FOR TIME

    def draw(self):
        """Draws the game state."""
        self.screen.fill(GRID_COLOR) # Fill screen with background color - DRAW BACKGROUND

        # Draw walls with thinner lines - DRAW WALLS
        wall_thickness = self.grid_size // 3  # Wall thickness reduced - WALL THICKNESS
        for y in range(len(MAZE)): # Iterate over rows - ITERATE
            for x in range(len(MAZE[y])): # Iterate over columns - ITERATE
                if MAZE[y][x] == 1: # If there is a wall - IS WALL
                    pygame.draw.rect(self.screen, WALL_COLOR, (x * self.grid_size, y * self.grid_size, wall_thickness, wall_thickness)) # Draw the wall - DRAW WALL

        # Draw Pac-Man (scaled down) - DRAW PACMAN
        pacman_size = self.grid_size // 2  # Smaller Pac-Man - PACMAN SIZE
        pygame.draw.rect(self.screen, PACMAN_COLOR, (self.pacman[0] * self.grid_size + self.grid_size // 4, # Draw the pacman - DRAW PACMAN
                                                     self.pacman[1] * self.grid_size + self.grid_size // 4, # Calculate top left corner of the rectangle - CALCULATE POSITION
                                                     pacman_size, pacman_size)) # Calculate the size of the rectangle - CALCULATE SIZE

        # Draw Ghosts - DRAW GHOSTS
        for ghost in self.ghosts: # Iterate over the ghosts - ITERATE
            pygame.draw.rect(self.screen, GHOST_COLOR, (ghost[0] * self.grid_size, ghost[1] * self.grid_size, self.grid_size, self.grid_size)) # Draw the ghost - DRAW GHOST

        # Draw Pellets - DRAW PELLETS
        for pellet in self.pellets: # Iterate over the pellets - ITERATE
            pygame.draw.rect(self.screen, PELLET_COLOR, (pellet[0] * self.grid_size, pellet[1] * self.grid_size, self.grid_size, self.grid_size)) # Draw the pellet - DRAW PELLET

        # Draw score - DRAW SCORE
        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR) # Render the score text - RENDER SCORE
        self.screen.blit(score_text, (10, 10)) # Draw the score text - DRAW SCORE

        pygame.display.flip() # Update the display - UPDATE DISPLAY

    def quit(self):
        """Quits the game."""
        pygame.quit() # Quit pygame - QUIT PYGAME

    def is_done(self):
        """Check if the game is over."""
        return self.game_over # Return game over status - RETURN GAME OVER STATUS

    def get_action_space(self):
        """Get the number of possible actions."""
        return self.action_size # Return the number of actions - RETURN NUMBER OF ACTIONS
