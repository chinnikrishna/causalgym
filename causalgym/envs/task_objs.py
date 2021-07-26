"""
Various objects used for building tasks
"""
# Imports
import pygame


class Bot(pygame.sprite.Sprite):
    def __init__(self, st_x, st_y, maxx, maxy):
        super(Bot, self).__init__()
        self.maxx = maxx
        self.maxy = maxy
        self.img = pygame.image.load('assets/bot.bmp')
        self.surf = pygame.Surface((st_x, st_y))
        self.rect = self.surf.get_rect()
        self.rect.x, self.rect.y = st_x, st_y
        self.rect.height = self.img.get_height()
        self.rect.width = self.img.get_width()
        self.speed = 1
        self.got_pass = False

    def update(self, pressed_key):
        """
        Updates bot position based on key presses
        """
        if pressed_key == pygame.K_LEFT:
            if self.rect.left > 0 + self.speed:
                self.rect.move_ip(-self.speed, 0)
        if pressed_key == pygame.K_RIGHT:
            if self.rect.right < self.maxx - self.speed:
                self.rect.move_ip(self.speed, 0)
        if pressed_key == pygame.K_UP:
            if self.rect.top > 0 + self.speed:
                self.rect.move_ip(0, -self.speed)
        if pressed_key == pygame.K_DOWN:
            if self.rect.bottom < self.maxy - self.speed:
                self.rect.move_ip(0, self.speed)

    def draw(self, screen):
        screen.blit(self.img, self.rect)


class Door(pygame.sprite.Sprite):
    def __init__(self, st_x, st_y, door_color='orange'):
        super(Door, self).__init__()
        if door_color == 'orange':
            self.img = pygame.image.load('assets/orange_door.bmp')
        elif door_color == 'green':
            self.img = pygame.image.load('assets/green_door.bmp')
        else:
            raise ValueError("Unknown door color")
        self.surf = pygame.Surface((st_x, st_y))
        self.rect = self.surf.get_rect()
        self.rect.x, self.rect.y = st_x, st_y
        self.rect.height = self.img.get_height()
        self.rect.width = self.img.get_width()

    def update(self, bot, pressed_key):
        """
        Checks interaction of bot with door when SPACE is pressed
        """
        if bot.rect.colliderect(self.rect):
            if pressed_key == pygame.K_SPACE:
                if bot.got_pass:
                    return 1
                else:
                    return 0
        else:
            return None

    def draw(self, screen):
        screen.blit(self.img, self.rect)


class Box(pygame.sprite.Sprite):
    def __init__(self, st_x, st_y, box_color='red', has_pass=False):
        super(Box, self).__init__()
        if box_color == 'red':
            self.box_open_img = pygame.image.load('assets/red_box_open.bmp')
            self.box_closed_img = pygame.image.load('assets/red_box_closed.bmp')
        elif box_color == 'green':
            self.box_open_img = pygame.image.load('assets/green_box_open.bmp')
            self.box_closed_img = pygame.image.load('assets/green_box_closed.bmp')
        elif box_color == 'blue':
            self.box_open_img = pygame.image.load('assets/blue_box_open.bmp')
            self.box_closed_img = pygame.image.load('assets/blue_box_closed.bmp')
        else:
            raise ValueError("Unknown box color")
        self.img = self.box_closed_img
        self.surf = pygame.Surface((st_x, st_y))
        self.rect = self.surf.get_rect()
        self.rect.x, self.rect.y = st_x, st_y
        self.rect.height = self.img.get_height()
        self.rect.width = self.img.get_width()
        self.has_pass = has_pass
        self.checked = False

    def update(self, bot, pressed_key):
        """
        Detects interaction with bot and updates Box state
        """
        if bot.rect.colliderect(self.rect):
            # Check if open action key is pressed
            if pressed_key == pygame.K_SPACE:
                self.img = self.box_open_img
                self.checked = True
                # Store pass in bot only when it doesnt have it
                if not bot.got_pass:
                    bot.got_pass = self.has_pass

    def draw(self, screen):
        screen.blit(self.img, self.rect)
