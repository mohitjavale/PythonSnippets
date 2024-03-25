import pygame

class PygameInput():

    def init_pyGameInput(self):
            pygame.init()
            self.screen = pygame.display.set_mode((100, 100))

    def get_pyGameInput(self):
        pygame.event.get()
        keys = pygame.key.get_pressed()
        cmd = [0,0,0]
        if keys[pygame.K_UP]:
            cmd[0]=1
        if keys[pygame.K_DOWN]:
            cmd[0]=-1
        if keys[pygame.K_LEFT]:
            cmd[2]=1
        if keys[pygame.K_RIGHT]:
            cmd[2]=-1
        if keys[pygame.K_COMMA]:
            cmd[1]=1
        if keys[pygame.K_PERIOD]:
            cmd[2]=-1
        return cmd
    

if __name__=='__main__':
    pgi = PygameInput()
    pgi.init_pyGameInput()
    while True:
        print(pgi.get_pyGameInput())