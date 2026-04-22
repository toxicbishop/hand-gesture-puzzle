import numpy as np
import random
import cv2

class Puzzle:
    def __init__(self, grid_size=3):
        self.grid_size = grid_size
        self.tiles = []
        self.order = []
        self.selected = None
        self.reference = None
        self._cached_combined = None

    def create(self, frame, grid_size=None):
        if grid_size:
            self.grid_size = grid_size

        h, w, _ = frame.shape
        th, tw = h // self.grid_size, w // self.grid_size

        self.tiles = []

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                tile = frame[i*th:(i+1)*th, j*tw:(j+1)*tw]
                tile = cv2.resize(tile, (tw, th))
                self.tiles.append(tile)

        n = len(self.tiles)
        self.order = list(range(n))

        rows = []
        for i in range(self.grid_size):
            row = np.hstack(self.tiles[i*self.grid_size:(i+1)*self.grid_size])
            rows.append(row)
        self.reference = np.vstack(rows)

        random.shuffle(self.order)
        self._cached_combined = None

    def combine(self):
        if self._cached_combined is not None:
            return self._cached_combined

        tiles = [self.tiles[i] for i in self.order]
        rows = []
        for i in range(self.grid_size):
            row = np.hstack(tiles[i*self.grid_size:(i+1)*self.grid_size])
            rows.append(row)
        self._cached_combined = np.vstack(rows)
        return self._cached_combined

    def get_index(self, x, y):
        col = int(x * self.grid_size)
        row = int(y * self.grid_size)
        col = max(0, min(col, self.grid_size - 1))
        row = max(0, min(row, self.grid_size - 1))
        return row * self.grid_size + col

    def swap(self, i, j):
        if i is not None and j is not None and i != j:
            self.order[i], self.order[j] = self.order[j], self.order[i]
            self._cached_combined = None

    def _tile_rect(self, idx, frame_w, frame_h):
        gs = self.grid_size
        th = frame_h // gs
        tw = frame_w // gs
        row = idx // gs
        col = idx % gs
        x1 = col * tw
        y1 = row * th
        return x1, y1, x1 + tw, y1 + th

    def draw_selected(self, frame, target_idx=None):
        h, w, _ = frame.shape

        if self.selected is not None:
            x1, y1, x2, y2 = self._tile_rect(self.selected, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (180, 255, 180), 1)

        if target_idx is not None and target_idx != self.selected:
            x1, y1, x2, y2 = self._tile_rect(target_idx, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

    def draw_reference(self, frame, max_size=160, margin=20):
        if self.reference is None:
            return

        h, w, _ = frame.shape
        rh, rw = self.reference.shape[:2]
        scale = max_size / max(rh, rw)
        new_w = int(rw * scale)
        new_h = int(rh * scale)
        thumb = cv2.resize(self.reference, (new_w, new_h))

        x = w - new_w - margin
        y = margin
        frame[y:y+new_h, x:x+new_w] = thumb
        cv2.rectangle(frame, (x-2, y-2), (x+new_w+2, y+new_h+2), (255, 255, 255), 2)

    def is_solved(self):
        return self.order == sorted(self.order)

    def get_solved_percentage(self):
        if not self.order:
            return 0
        correct = sum(1 for i, v in enumerate(self.order) if i == v)
        return (correct / len(self.order)) * 100
