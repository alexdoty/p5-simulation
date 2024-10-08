from __future__ import annotations
from abc import abstractmethod
from typing import Annotated, override

from pygame import gfxdraw
from pygame.font import Font
from pygame.rect import Rect
from pygame.surface import Surface

colors = {
    "black": (25, 25, 25),
    "dgray": (206, 208, 208),
    "lgray": (230, 232, 230),
    "white": (255, 255, 255),
    "orange": (241, 80, 37),
}


def layersort(d: Drawable) -> bool:
    return d.layer == 1


class Drawable:
    has_font: bool
    pos: tuple[int, int]
    layer: int

    def __init__(self, has_font, pos, layer) -> None:
        self.has_font = has_font
        self.pos = pos
        self.layer = layer

    @classmethod
    def to_manager(cls, manager: DrawManager, args: tuple):
        print(args)
        d = cls(*args)
        manager.add(d)

    @abstractmethod
    def draw(self, canvas: Surface):
        pass

    @abstractmethod
    def setup(self, *args):
        pass


class Node(Drawable):
    imp: complex | None
    id: int
    ridtext: Surface
    ridrect: Rect
    rimptext: Surface
    rimprect: Rect
    sizes: tuple[int, int] = (40, 38)

    def __init__(self, pos, imp, id, has_font=True, layer=1) -> None:
        super().__init__(has_font, pos, layer)
        self.imp = imp
        self.id = id

    @override
    def draw(self, canvas: Surface):
        gfxdraw.filled_circle(
            canvas, self.pos[0], self.pos[1], self.sizes[0], colors["lgray"]
        )
        gfxdraw.aacircle(
            canvas, self.pos[0], self.pos[1], self.sizes[0], colors["lgray"]
        )
        gfxdraw.filled_circle(
            canvas, self.pos[0], self.pos[1], self.sizes[1], colors["white"]
        )
        canvas.blit(self.ridtext, self.ridrect)
        canvas.blit(self.rimptext, self.rimprect)

    @override
    def setup(self, *args):
        font = args[0]
        if len(args) >= 2:
            self.sizes = args[1]

        self.ridtext = font.render(str(self.id), True, colors["black"], None)
        ridrect = self.ridtext.get_rect()
        print("Pos:")
        print(self.pos)
        ridrect.center = self.pos
        self.ridrect = ridrect

        if self.imp is not None:
            self.rimptext = font.render(str(self.imp), True, colors["white"], None)
            rimprect = self.rimptext.get_rect()
            rimprect.center = (self.pos[0] - 30, self.pos[1] + 70)
            self.rimprect = rimprect


class Edge(Drawable):
    nodes: tuple[tuple[int, int], tuple[int, int]]
    imp: complex | None
    rimptext: Surface
    rimprect: Rect

    def __init__(self, nodes, imp, has_font=True, layer=0) -> None:
        super().__init__(has_font, None, layer)
        self.nodes = nodes
        self.imp = imp

    @override
    def draw(self, canvas: Surface):
        gfxdraw.line(canvas, self.nodes[0][0], self.nodes[0][1], self.nodes[1][0], self.nodes[1][1], colors["orange"])
        canvas.blit(self.rimptext, self.rimprect)

    @override
    def setup(self, *args):
        font = args[0]
        self.pos = ((self.nodes[0][0] + self.nodes[1][0])//2, (self.nodes[0][1] + self.nodes[1][1])//2)
        if self.imp is not None:
            self.rimptext = font.render(str(self.imp), True, colors["white"], None)
            rimprect = self.rimptext.get_rect()
            rimprect.center = (self.pos[0] - 30, self.pos[1] + 70)
            self.rimprect = rimprect



class DrawManager:
    # Assumed 0 is bg and 1 is fg
    layers: tuple[Surface, Surface]
    font: Font
    drawables: list[Drawable] = []

    def __init__(self, layer0: Surface, layer1: Surface, font: Font) -> None:
        self.layers = (layer0, layer1)
        self.font = font

    def add(self, drawable: Drawable):
        self.drawables.append(drawable)
        # self.drawables = sorted(self.drawables, key=layersort)

    def setup(self):
        self.drawables = sorted(self.drawables, key=layersort)
        for d in self.drawables:
            if d.has_font:
                d.setup(self.font)
                continue
            d.setup()

    def draw(self):
        for d in self.drawables:
            match d.layer:
                case 0:
                    d.draw(self.layers[0])
                case 1:
                    d.draw(self.layers[1])
                case _:
                    raise ValueError
