import time
import pygame
import numpy as np
from PIL import Image
import os

from gamefunctions import *

pygame.init()

grid_size = 28
cell_size = 14
display_size = grid_size * cell_size
window_width = display_size * 2 + 20
window_height = display_size + 60
button_height = 40

screen = pygame.display.set_mode((window_width, window_height))

black = (0, 0, 0)
white = (255, 255, 255)
gray = (200, 200, 200)
blue = (0, 0, 255)
red = (255, 0, 0)

drawing_matrix = np.zeros((grid_size, grid_size))

def draw_grid(offset_x, offset_y):
    for i in range(grid_size):
        for j in range(grid_size):
            color = black if drawing_matrix[i][j] == 1 else white
            pygame.draw.rect(screen, color, (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size))
            pygame.draw.rect(screen, gray, (offset_x + j * cell_size, offset_y + i * cell_size, cell_size, cell_size), 1)

def draw_buttons():
    pygame.draw.rect(screen, blue, (display_size + 30, 20, display_size, button_height))
    font = pygame.font.Font(None, 36)
    text_test = font.render("Тест", True, white)
    screen.blit(text_test, (display_size + 30 + display_size // 2 - text_test.get_width() // 2, 20 + button_height // 2 - text_test.get_height() // 2))

    pygame.draw.rect(screen, red, (display_size + 30, 80, display_size, button_height))
    text_clear = font.render("Очистить", True, white)
    screen.blit(text_clear, (display_size + 30 + display_size // 2 - text_clear.get_width() // 2, 80 + button_height // 2 - text_clear.get_height() // 2))

    pygame.draw.rect(screen, 'green', (display_size + 30, 140, display_size, button_height))
    text_correct = font.render("Правильно", True, white)
    screen.blit(text_correct, (display_size + 30 + display_size // 2 - text_correct.get_width() // 2, 140 + button_height // 2 - text_correct.get_height() // 2))

def save_image(matrix, filename="drawing.png"):
    img = Image.fromarray(np.uint8((1 - matrix) * 255), 'L')
    img.save(filename)

def save_image_in_dataset(matrix, filename):
    img = Image.fromarray(np.uint8((1 - matrix)*255), 'L')
    img.save("dataset/" + filename)

def show_message(text):
    font = pygame.font.Font(None, 36)
    message = font.render(text, True, blue)
    screen.blit(message, (display_size + 30 + display_size // 2 - message.get_width() // 2, 260))

def draw_cross(i, j):
    if 0 <= i < grid_size and 0 <= j < grid_size:
        drawing_matrix[i][j] = 1  
    if 0 <= i - 1 < grid_size and 0 <= j < grid_size:
        drawing_matrix[i - 1][j] = 1
    if 0 <= i + 1 < grid_size and 0 <= j < grid_size:
        drawing_matrix[i + 1][j] = 1  
    if 0 <= i < grid_size and 0 <= j - 1 < grid_size:
        drawing_matrix[i][j - 1] = 1
    if 0 <= i < grid_size and 0 <= j + 1 < grid_size:
        drawing_matrix[i][j + 1] = 1  


running = True
drawing = False
show_test_message = False
pr = [0, 0]
while running:
    rest = ""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_y < display_size and mouse_x < display_size:
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if display_size + 30 <= mouse_x <= display_size * 2 + 30:
                if 20 <= mouse_y <= 20 + button_height: 
                    save_image(drawing_matrix)
                    show_test_message = True
                    rest = list(map(float, f"{res()[0]}".replace("[", "").replace("]", "").split()))
                    pr = [rest.index(max(rest)), max(rest)]
                elif 80 <= mouse_y <= 80 + button_height:
                    drawing_matrix = np.zeros((grid_size, grid_size))
                    show_test_message = False
                elif 140 <= mouse_y <= 140 + button_height:
                    files = os.listdir("dataset")
                    png_files = [f for f in files if (f.endswith(".png") and f.startswith(str(pr[0])))]
                    sorted_files = sorted(png_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    last_file = sorted_files[-1]
                    save_image_in_dataset(drawing_matrix, f"{pr[0]}_{int(last_file.split('_')[1].split('.')[0])+1}.png")
                    add_txt_to_dataset(f"dataset/{pr[0]}_{int(last_file.split('_')[1].split('.')[0])+1}.png", 
                                       f"dataset/{pr[0]}_{int(last_file.split('_')[1].split('.')[0])+1}.txt", str(pr[0]))
                    print("Correct", f"dataset/{pr[0]}_{int(last_file.split('_')[1].split('.')[0])+1}.txt")
        elif event.type == pygame.MOUSEMOTION and drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_y < display_size and mouse_x < display_size: 
                j, i = mouse_x // cell_size, mouse_y // cell_size
                draw_cross(i, j)  

    screen.fill(white)
    draw_grid(0, 0)
    draw_buttons()
    if show_test_message:
        show_message(f"Возможно это {pr[0]}")
    pygame.display.flip()

pygame.quit()