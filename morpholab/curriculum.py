"""Completed-episode gates for the optional staged locomotion curriculum."""
STAGES = (
    {"name": "Balance", "preset": "balance", "speed_fraction": 0, "randomization": 0},
    {"name": "Slow walk", "preset": "velocity", "speed_fraction": .25, "randomization": .25},
    {"name": "Walk", "preset": "velocity", "speed_fraction": .6, "randomization": .6},
    {"name": "Target speed", "preset": "velocity", "speed_fraction": 1, "randomization": 1},
)


class LearningCurriculum:
    def __init__(self, target_speed):
        self.target_speed = target_speed
        self.index = 0
        self.episodes = []

    @property
    def stage(self):
        return STAGES[self.index]

    def record(self, *, fell, speed_error):
        self.episodes.append({"fell": fell, "speed_error": speed_error})
        self.episodes = self.episodes[-12:]

    def advance(self):
        if self.index == len(STAGES) - 1 or len(self.episodes) < 12:
            return False
        survival = sum(not e["fell"] for e in self.episodes) / len(self.episodes)
        error = sum(e["speed_error"] for e in self.episodes) / len(self.episodes)
        tolerance = max(.15, self.target_speed * self.stage["speed_fraction"] * .35)
        if survival < .8 or error > tolerance:
            return False
        self.index += 1
        self.episodes = []
        return True

    def snapshot(self):
        return {"index": self.index, "name": self.stage["name"], "episodes": len(self.episodes),
                "speed": self.target_speed * self.stage["speed_fraction"],
                "randomization": self.stage["randomization"]}
