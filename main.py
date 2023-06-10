import pygame

# constants
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
LEFT = 1
SCROLL = 2
RIGHT = 3

LANDMARK_IMG = 'landmark.png'


class Landmark(pygame.sprite.Sprite):
    def __init__(self, row, column,  image, trans_color=BLACK):
        super(Landmark, self).__init__()
        self.image = pygame.image.load(image)
        self.image.set_colorkey(trans_color)
        self.__row = row
        self.__column = column
        self.rect = self.image.get_rect()
        self.rect.x = column - self.rect.width / 2
        self.rect.y = row - 40


def init_screen(width=WINDOW_WIDTH, height=WINDOW_HEIGHT):
    size = (width, height)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Game")
    return screen


def fill_screen(screen, color):
    screen.fill((color[0], color[1], color[2]))
    pygame.display.flip()


def main():
    pygame.init()
    scr = init_screen()
    fill_screen(scr, WHITE)
    print('Place landmarks using left button.\n' +
          'Press right button to draw an obstacle.\n ')
    finish = False
    landmarks_list = pygame.sprite.Group()
    while not finish:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN \
                    and event.button == LEFT:
                pos = pygame.mouse.get_pos()
                landmark = Landmark(column=pos[0], row=pos[1], image=LANDMARK_IMG)
                landmarks_list.add(landmark)
                landmarks_list.draw(scr)
                pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONDOWN \
                    and event.button == RIGHT:
                finish = True

    pygame.quit()


if __name__ == '__main__':
    main()