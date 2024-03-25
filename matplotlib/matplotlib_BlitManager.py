import numpy as np
import matplotlib.pyplot as plt

class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()

if __name__=='__main__':
    
    # Generate data
    x = []
    y = []

    # make a new figure
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set(xlim=(-5,5), ylim=(-5,5))

    # Static Artists
    ax.axvline(x=0, c="grey", label="x=0")
    ax.axhline(y=0, c="grey", label="y=0")


    # make dynamic artist
    sin_line, = ax.plot(x, y)
    
    bm = BlitManager(fig.canvas)
    bm.add_artist(sin_line)
    
    plt.show(block=False)
    plt.pause(0.1)
    
    i = 0

    while True:
        i += 1
        x = np.linspace(-4, 4, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        sin_line.set_data(x,y) 
        bm.update()
    
    plt.show()
