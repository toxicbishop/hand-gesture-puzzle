import json
import os

class ScoresManager:
    def __init__(self, filename="scores.json"):
        self.filename = filename
        self.scores = self.load_scores()

    def load_scores(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def get_best_time(self, grid_size):
        key = str(grid_size)
        return self.scores.get(key, float('inf'))

    def update_score(self, grid_size, time_taken):
        key = str(grid_size)
        current_best = self.get_best_time(grid_size)
        if time_taken < current_best:
            self.scores[key] = round(time_taken, 2)
            self.save_scores()
            return True
        return False

    def save_scores(self):
        with open(self.filename, 'w') as f:
            json.dump(self.scores, f, indent=4)
