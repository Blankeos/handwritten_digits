import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# CONSTANTS
WINDOW_SIZE_X = 640
WINDOW_SIZE_Y = 480
WHITE = (255,255,255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ALLOW_IMAGE_SAVE = False
MODEL = load_model("bestmodel.h5")
LABEL_NAMES = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
BOUNDARYINC = 5

# INITIALIZATION
pygame.init()
pygame.display.set_caption("Digit Board")

DISPLAY_SURFACE = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
FONT = pygame.font.Font("freesansbold.ttf", 18)

# RUNTIME VARIABLES
iswriting = False
number_xcord = []
number_ycord = []
image_count = 1
PREDICT = True

# GAME LOOP
while True:
    for event in pygame.event.get():
        if (event.type == QUIT):
            pygame.quit()
            sys.exit()

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOW_SIZE_X, number_xcord[-1]+BOUNDARYINC)

            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1]+BOUNDARYINC, WINDOW_SIZE_X)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAY_SURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if ALLOW_IMAGE_SAVE:
                cv2.imwrite("image.png")
                image_count += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28,28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image, (28,28))/255

                label = str(LABEL_NAMES[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()

                # Centering algorithm
                imageSizeX = rect_max_x - rect_min_x
                textSizeX = textRecObj.size[0]
                xOffset = (imageSizeX - textSizeX) / 2
                textRecObj.left, textRecObj.top = rect_min_x + xOffset, rect_max_y

                # top line
                pygame.draw.line(DISPLAY_SURFACE, RED, (rect_min_x, rect_min_y), (rect_max_x, rect_min_y), 3)
                # bottom line
                pygame.draw.line(DISPLAY_SURFACE, RED, (rect_min_x, rect_max_y), (rect_max_x, rect_max_y), 3)
                # left line
                pygame.draw.line(DISPLAY_SURFACE, RED, (rect_min_x, rect_min_y), (rect_min_x, rect_max_y), 3)
                # right line
                pygame.draw.line(DISPLAY_SURFACE, RED, (rect_max_x, rect_min_y), (rect_max_x, rect_max_y), 3)

                DISPLAY_SURFACE.blit(textSurface, textRecObj)



        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAY_SURFACE, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)
        
        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAY_SURFACE.fill(BLACK)
    
    pygame.display.update()

    

