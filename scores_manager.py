import json
import os

SCORES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores.json")


class ScoresManager:
    def __init__(self, filename=SCORES_PATH):
        self.filename = filename
        self.scores = self.load_scores()

    def load_scores(self):
        if not os.path.exists(self.filename):
            return {}
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: Could not load scores from {self.filename}: {e}")
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
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.scores, f, indent=4)
        except OSError as e:
            print(f"WARNING: Could not save scores to {self.filename}: {e}")
