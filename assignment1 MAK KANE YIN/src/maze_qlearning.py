import pygame
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Initialize pygame
pygame.init()

# Game constant definitions
WIDTH, HEIGHT = 600, 600
CELL_SIZE = 60
GRID_SIZE = 10
FPS = 30

# Color definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)    # Agent
GREEN = (0, 255, 0)  # Goal
BLUE = (0, 0, 255)   # Start
GRAY = (128, 128, 128)# Obstacle

# Q-Learning Agent Class
class QLearningAgent:
    def __init__(self, grid_size, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.grid_size = grid_size
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = 0.05 # Minimum exploration rate
        self.epsilon_decay = 0.995 # Exploration rate decay
        
        # Initialize Q-table: (x, y) -> [up, down, left, right]
        self.q_table = np.zeros((grid_size, grid_size, 4))
        
        # Action definitions
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
    
    def choose_action(self, state):
        """ε-greedy strategy for action selection"""
        x, y = state
        # Exploration: randomly select action
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])
        # Exploitation: select action with maximum Q-value
        else:
            return np.argmax(self.q_table[x, y])
    
    def learn(self, state, action, reward, next_state, done):
        """Q-Learning update formula"""
        x, y = state
        nx, ny = next_state
        
        # Q-Learning core formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        old_value = self.q_table[x, y, action]
        next_max = np.max(self.q_table[nx, ny]) if not done else 0
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[x, y, action] = new_value
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Maze Game Environment Class
class MazeGame:
    def __init__(self):
        # Set up game window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Q-Learning Maze Escape")
        self.clock = pygame.time.Clock()
        
        # Create maze (1=obstacle, 0=passable)
        self.maze = self._create_maze()
        
        # Define start and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (9, 9)
        
        # Initialize agent
        self.agent = QLearningAgent(GRID_SIZE)
        self.agent_pos = list(self.start_pos)
        
        # Training-related variables
        self.episodes = 1000  # Number of training episodes
        self.current_episode = 0
        self.total_reward = 0
        self.rewards_history = []  # Record rewards per episode
        
        # UI display variables
        self.font = pygame.font.SysFont(None, 24)
    
    def _create_maze(self):
        """Create maze map (random obstacles)"""
        maze = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Add obstacles (ensure start and goal are passable)
        obstacles = [
            (1, 1), (1, 2), (1, 3),
            (2, 5), (3, 5), (4, 5),
            (5, 2), (6, 2), (7, 2),
            (7, 7), (7, 8), (8, 7)
        ]
        
        for x, y in obstacles:
            maze[x, y] = 1
        
        return maze
    
    def get_reward(self, next_pos):
        """Reward function"""
        x, y = next_pos
        
        # Reach goal
        if (x, y) == self.goal_pos:
            return 100, True
        
        # Hit obstacle or boundary
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or self.maze[x, y] == 1:
            return -10, False
        
        # Normal movement (small penalty to encourage shortest path)
        return -1, False
    
    def step(self, action):
        """Execute action and return new state, reward, and done flag"""
        x, y = self.agent_pos
        dx, dy = self.agent.actions[action]
        next_x, next_y = x + dx, y + dy
        
        # Get reward and done status
        reward, done = self.get_reward((next_x, next_y))
        
        # Update agent position (if valid)
        if 0 <= next_x < GRID_SIZE and 0 <= next_y < GRID_SIZE and self.maze[next_x, next_y] == 0:
            self.agent_pos = [next_x, next_y]
        
        return tuple(self.agent_pos), reward, done
    
    def draw_maze(self):
        """Draw maze on screen"""
        self.screen.fill(WHITE)
        
        # Draw grid
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Draw obstacle
                if self.maze[x, y] == 1:
                    pygame.draw.rect(self.screen, GRAY, rect)
                # Draw start position
                elif (x, y) == self.start_pos:
                    pygame.draw.rect(self.screen, BLUE, rect)
                # Draw goal position
                elif (x, y) == self.goal_pos:
                    pygame.draw.rect(self.screen, GREEN, rect)
                # Draw empty cell
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, BLACK, rect, 1)
        
        # Draw agent
        agent_rect = pygame.Rect(
            self.agent_pos[1]*CELL_SIZE + 5,
            self.agent_pos[0]*CELL_SIZE + 5,
            CELL_SIZE - 10,
            CELL_SIZE - 10
        )
        pygame.draw.circle(self.screen, RED, agent_rect.center, CELL_SIZE//2 - 5)
        
        # Display text information
        episode_text = self.font.render(f"Episode: {self.current_episode}/{self.episodes}", True, BLACK)
        reward_text = self.font.render(f"Total Reward: {self.total_reward:.1f}", True, BLACK)
        epsilon_text = self.font.render(f"Epsilon: {self.agent.epsilon:.2f}", True, BLACK)
        
        self.screen.blit(episode_text, (10, HEIGHT - 80))
        self.screen.blit(reward_text, (10, HEIGHT - 50))
        self.screen.blit(epsilon_text, (10, HEIGHT - 20))
        
        pygame.display.update()
    
    def train(self):
        """Train the agent"""
        running = True
        for episode in range(self.episodes):
            self.current_episode = episode + 1
            self.agent_pos = list(self.start_pos)
            self.total_reward = 0
            done = False
            
            while not done:
                # Handle quit event
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        return
                
                # 1. Get current state
                current_state = tuple(self.agent_pos)
                
                # 2. Choose action
                action = self.agent.choose_action(current_state)
                
                # 3. Execute action
                next_state, reward, done = self.step(action)
                
                # 4. Learn and update Q-table
                self.agent.learn(current_state, action, reward, next_state, done)
                
                # 5. Update total reward
                self.total_reward += reward
                
                # 6. Draw game screen
                self.draw_maze()
                self.clock.tick(FPS)
            
            # Record reward for this episode
            self.rewards_history.append(self.total_reward)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}, Avg Reward: {np.mean(self.rewards_history[-10:]):.2f}, Epsilon: {self.agent.epsilon:.2f}")
        
        # Plot reward curve after training
        self.plot_rewards()
        
        # Show optimal path
        self.show_optimal_path()
    
    def plot_rewards(self):
        """Plot reward change curve"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards_history)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Q-Learning Training Progress')
        plt.grid(True)
        # Save to outputs folder (create folder if it doesn't exist)
        import os
        os.makedirs('../outputs', exist_ok=True)  # Go up 1 level to root, then into outputs/
        plt.savefig('../outputs/training_rewards.png')
        plt.show()
    
    def show_optimal_path(self):
        """Show optimal path after training"""
        print("\nShowing optimal path...")
        self.agent_pos = list(self.start_pos)
        self.agent.epsilon = 0  # Disable exploration, use only optimal policy
        
        path = [tuple(self.agent_pos)]
        done = False
        steps = 0
        
        while not done and steps < 100:  # Prevent infinite loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            current_state = tuple(self.agent_pos)
            action = self.agent.choose_action(current_state)
            next_state, _, done = self.step(action)
            
            path.append(next_state)
            self.draw_maze()
            self.clock.tick(5)  # Slow motion for demonstration
            steps += 1
        
        print(f"Optimal path steps: {steps}")
        print(f"Path: {path}")

# Main function
def main():
    game = MazeGame()
    
    # Start training
    print("Starting Q-Learning agent training...")
    game.train()
    
    # Keep window open
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        game.draw_maze()
        game.clock.tick(FPS)

if __name__ == "__main__":
    main()
