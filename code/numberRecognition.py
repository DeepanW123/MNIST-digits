import pygame
import numpy as np
import tensorflow as tf

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 560, 560  # 20x scale for 28x28 grid
GRID_SIZE = 28
CELL_SIZE = WIDTH // GRID_SIZE
WHITE = (0, 0, 0)
BLACK = (255, 255, 255)
RED = (255, 0, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH + 200, HEIGHT))
pygame.display.set_caption("Digit Guessing App")
font = pygame.font.SysFont("Arial", 40)

# Initialize grid
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

# Load pre-trained model
model = tf.keras.models.load_model('code/best_model.h5')


# Predict digit after smoothing
def predict_digit(grid):
    img = grid.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img)
    print(prediction)
    return np.argmax(prediction)

# Draw grid lines
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y))

# Draw cells
def draw_cells():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] > 0:
                pygame.draw.rect(
                    screen,
                    BLACK,
                    (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )

# Main loop
running = True
prediction = None
while running:
    screen.fill(WHITE)
    draw_cells()
    draw_grid()

    # Display prediction
    pygame.draw.rect(screen, RED, (WIDTH, 0, 200, HEIGHT))
    text = font.render("Guess:", True, WHITE)
    screen.blit(text, (WIDTH + 40, 50))
    if prediction is not None:
        pred_text = font.render(str(prediction), True, WHITE)
        screen.blit(pred_text, (WIDTH + 80, 120))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Drawing with mouse (brush effect)
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            if x < WIDTH:
                col = x // CELL_SIZE
                row = y // CELL_SIZE

                # Apply brush effect
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if 0 <= row + i < GRID_SIZE and 0 <= col + j < GRID_SIZE:
                            grid[row + i][col + j] = 255

        # Key events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                if np.sum(grid) == 0:
                    prediction = "Draw a digit!"
                else:
                    prediction = predict_digit(grid)
            if event.key == pygame.K_c:
                grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
                prediction = None

    pygame.display.flip()

pygame.quit()
