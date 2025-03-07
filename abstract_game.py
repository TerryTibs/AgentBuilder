# Import the Abstract Base Class (ABC) module for creating abstract classes - USED FOR DEFINING ABSTRACT CLASSES
from abc import ABC, abstractmethod

# ---------------- Abstract Game Class ---------------- #
class AbstractGame(ABC):
    """
    An abstract base class for defining game environments.

    This class acts as a template for specific game implementations
    (e.g., Snake, Pacman). Any game class that inherits from AbstractGame
    must define all the abstract methods below.
    """

    @abstractmethod
    def __init__(self):
        """
        Initializes the game environment.

        This method must be implemented in subclasses to set up necessary
        game parameters, such as screen size, game objects, or initial states.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def reset(self):
        """
        Resets the game to its initial state.

        This method should be implemented in subclasses to restart the game
        without requiring a new instance of the game object.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def step(self, action):
        """
        Takes an action and updates the game state accordingly.

        Args:
            action: The action chosen by the AI or player.

        Returns:
            - The new game state (usually a NumPy array representation).
            - The reward received from the action.
            - A done flag indicating whether the game is over.

        This method is crucial for reinforcement learning, as the agent
        learns by interacting with the game environment through steps.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def get_state(self):
        """
        Returns the current state of the game.

        The state should be formatted as a NumPy array (or similar structure)
        that the AI can process for decision-making.

        Returns:
            - The current state of the game in a structured format.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def is_done(self):
        """
        Checks whether the game has ended.

        Returns:
            - A boolean value indicating if the game is over.

        This helps the AI determine when to reset and start a new episode.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def get_action_space(self):
        """
        Returns the number of possible actions in the game.

        This allows the AI to understand what actions it can take.

        Returns:
            - The size of the action space (e.g., 4 for up, down, left, right in Snake).
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def draw(self):
        """
        Renders the current game state on the screen.

        This method should handle all game graphics, such as drawing the
        game board, player, enemies, and other elements.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION

    @abstractmethod
    def quit(self):
        """
        Shuts down the game environment properly.

        This ensures that all game processes are closed and resources
        are freed when the game is no longer needed.
        """
        pass # MUST BE IMPLEMENTED BY SUBCLASSES - NO DEFAULT IMPLEMENTATION
