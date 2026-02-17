import time


class fps:
    def __init__(self, average_over=100):
        self.average_over = average_over
        self.times_array = []

    def start(self):
        self.tic = time.time()

    def update(self):
        toc = time.time()
        self.times_array.append(toc - self.tic)

    def get_time(self):
        return round(sum(self.times_array[-self.average_over :]) / self.average_over, 4)

    def get_unrounded_time(self):
        return sum(self.times_array[-self.average_over :]) / self.average_over

    def get_fps(self):
        return round(1 / self.get_unrounded_time(), 4)

    def get_total_time(self):
        return round(sum(self.times_array) / len(self.times_array), 4)

    def get_total_fps(self):
        return round(1 / self.get_total_time(), 4)
