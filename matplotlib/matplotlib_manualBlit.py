
import numpy as np
import matplotlib.pyplot as plt
import time

class BlitClass():

    def plt_init(self):

        self.fig, self.ax = plt.subplots()

        self.ax.set_aspect('equal')
        self.ax.set(xlim=(-1000,1000), ylim=(-1000,1000))

        # Static Artists
        self.ax.axvline(x=0, c="grey", label="x=0")
        self.ax.axhline(y=0, c="grey", label="y=0")

        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        # Dynamic Artists
        self.line, = plt.plot([],[], c='black')
        self.pt = plt.scatter([],[], c='black')


    def plt_blit(self, i=0):
        self.fig.canvas.restore_region(self.bg)

        # Update Dynamic Artists Data
        self.line.set_xdata(0)
        self.line.set_ydata(0)
        self.pt.set_offsets([i,i])

        # Draw Dynamic Artists
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.pt)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()


if __name__=='__main__':
    b = BlitClass()
    b.plt_init()

    for i in range(1000):
        b.plt_blit(i)








