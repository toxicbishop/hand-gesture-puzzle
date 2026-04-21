import numpy as np
import random
import cv2

class Puzzle:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.tiles = []
        self.selected = None

    def create(self, frame, grid_size=None):
        if grid_size:
            self.grid_size = grid_size
            
        h, w, _ = frame.shape
        th, tw = h // self.grid_size, w // self.grid_size

        self.tiles = []
        self.original_tiles = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                tile = frame[i*th:(i+1)*th, j*tw:(j+1)*tw]
                # Ensure tile is the right size (handle rounding)
                tile = cv2.resize(tile, (tw, th))
                self.tiles.append(tile)
                self.original_tiles.append(tile.copy())

        random.shuffle(self.tiles)

    def combine(self):
        rows = []
        for i in range(self.grid_size):
            row = np.hstack(self.tiles[i*self.grid_size:(i+1)*self.grid_size])
            rows.append(row)
        return np.vstack(rows)

    def get_index(self, x, y):
        col = int(x * self.grid_size)
        row = int(y * self.grid_size)
        col = max(0, min(col, self.grid_size - 1))
        row = max(0, min(row, self.grid_size - 1))
        return row * self.grid_size + col
    def swap(self, i, j):
        if i is not None and j is not None and i != j:
            self.tiles[i], self.tiles[j] = self.tiles[j], self.tiles[i]

    def draw_selected(self, frame):
        if self.selected is not None:
            gs = self.grid_size
            h, w, _ = frame.shape

            th = h // gs
            tw = w // gs

            row = self.selected // gs
            col = self.selected % gs

            x1 = col * tw
            y1 = row * th
            x2 = x1 + tw
            y2 = y1 + th

            # Glow effect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (180, 255, 180), 1)
    def is_solved(self):
        for i in range(len(self.tiles)):
            if not np.array_equal(self.tiles[i], self.original_tiles[i]):
                return False
        return True

    def get_solved_percentage(self):
        correct = 0
        for i in range(len(self.tiles)):
            if np.array_equal(self.tiles[i], self.original_tiles[i]):
                correct += 1
        return (correct / len(self.tiles)) * 100