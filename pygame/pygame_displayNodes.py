import pygame
import numpy as np
import sys

class PyGameGUI():
	def __init__(self):
		pygame.init()
		self.clock = pygame.time.Clock()
		self.screen = pygame.display.set_mode((500, 500))
		self.fps = 60
		self.timestep = 1/self.fps
		self.scaling_factor = 1
		self.timer = 0


	def display_nodes(self, node_list):
		# bg colour
		self.screen.fill((255,255,255))	

		for node in node_list:
			pygame.draw.circle(self.screen,(0,255,0),(node.x, node.y), radius=4)

		pygame.display.flip()
		self.clock.tick(self.fps)
		self.timer += self.timestep


class Node():
	def __init__(self):
		self.x = np.random.randint(0,500)
		self.y = np.random.randint(0,500)
		pass

if __name__=='__main__':
	pgg = PyGameGUI()

	node_list = []
	for _ in range(20):
		node = Node()
		node_list.append(node)

	while True:
		pgg.display_nodes(node_list)

		for event in pygame.event.get():  
			if event.type == pygame.QUIT:  
				sys.exit()

