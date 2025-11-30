import pygame
import numpy as np
import time
import os
from tensorflow.keras.models import load_model

CAR_IMAGE_PATH = "C:\\Users\\reddy\\Desktop\\Mini_project\\car.jpg"
MODEL_PATH = "model/eeg_attention_model.keras"

STATE_NAMES = ["Focused", "Unfocused", "Drowsy"]
speed_map = {0: 16, 1: 6, 2: 0}
state_speed_map = {"Focused": 120, "Unfocused": 40, "Drowsy": 0}

def play_game(X_test, model):
    pygame.init()
    WIDTH, HEIGHT = 600, 400
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("EEG Attention Car Game")

    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 28)
    clock = pygame.time.Clock()

    ROAD_WIDTH = 220
    ROAD_X = (WIDTH - ROAD_WIDTH) // 2
    LANE_COLOR = (120, 120, 120)
    GRASS_COLOR = (40, 150, 40)

    CAR_WIDTH, CAR_HEIGHT = 65, 44
    car_x = ROAD_X + (ROAD_WIDTH - CAR_WIDTH) // 2
    car_y = HEIGHT - CAR_HEIGHT - 20

    car_image = pygame.image.load(CAR_IMAGE_PATH) if os.path.exists(CAR_IMAGE_PATH) else None
    if car_image:
        car_image = pygame.transform.scale(car_image, (CAR_WIDTH, CAR_HEIGHT))

    LINE_WIDTH, LINE_HEIGHT = 8, 40
    LINE_COLOR = (255, 255, 255)
    NUM_LINES = 6
    line_positions = [i * (HEIGHT // NUM_LINES) for i in range(NUM_LINES)]

    running = True
    game_over = False
    game_over_time = None
    drowsy_count = 0
    DROWSY_THRESHOLD = 10

    i = 0
    total_samples = len(X_test)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get next EEG test sample
        sample = X_test[i].reshape(1, X_test.shape[1], X_test.shape[2])
        prediction = model.predict(sample, verbose=0)
        state = np.argmax(prediction)
        state_text = STATE_NAMES[state]
        speed = speed_map[state]

        print(f"Predicted State: {state_text}")

        if state == 2:
            drowsy_count += 1
        else:
            drowsy_count = 0

        if drowsy_count >= DROWSY_THRESHOLD and not game_over:
            game_over = True
            game_over_time = time.time()

        if not game_over:
            for j in range(len(line_positions)):
                line_positions[j] += speed
                if line_positions[j] > HEIGHT:
                    line_positions[j] -= HEIGHT

        screen.fill(GRASS_COLOR)
        pygame.draw.rect(screen, LANE_COLOR, (ROAD_X, 0, ROAD_WIDTH, HEIGHT))
        lane_x = ROAD_X + ROAD_WIDTH // 2 - LINE_WIDTH // 2
        for y in line_positions:
            pygame.draw.rect(screen, LINE_COLOR, (lane_x, int(y), LINE_WIDTH, LINE_HEIGHT))

        if car_image:
            screen.blit(car_image, (car_x, car_y))
        else:
            pygame.draw.rect(screen, (180, 180, 220), (car_x, car_y, CAR_WIDTH, CAR_HEIGHT))

        info = font.render(f"State: {state_text}", True, (255, 255, 255))
        screen.blit(info, (20, 20))
        speed_txt = small_font.render(f"Speed: {state_speed_map[state_text]} km/h", True, (255, 255, 255))
        screen.blit(speed_txt, (WIDTH - 180, 30))

        if game_over:
            over_txt = font.render("GAME OVER!", True, (255, 50, 50))
            screen.blit(over_txt, (WIDTH // 2 - 120, HEIGHT // 2 - 50))
            if time.time() - game_over_time > 2:
                game_over = False
                drowsy_count = 0

        pygame.display.flip()
        clock.tick(10)  # slower for EEG sample rate simulation
        i = (i + 1) % total_samples

    pygame.quit()
