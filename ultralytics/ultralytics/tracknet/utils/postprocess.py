import csv
import cv2
import threading
from queue import Queue

class PostprocessSaver:
    def __init__(self, num_workers=2):
        self.queue = Queue()
        self.workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
            self.workers.append(t)

    def worker(self):
        while True:
            task = self.queue.get()
            try:
                if task['type'] == 'csv':
                    self._save_csv(task['path'], task['rows'])
                elif task['type'] == 'image':
                    self._save_image(task['path'], task['image'])
            except Exception as e:
                print(f"[PostprocessSaver] Error during {task['type']} saving: {e}")
            finally:
                self.queue.task_done()

    def _save_csv(self, path, rows):
        with open(path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Frame', 'Visibility', 'X', 'Y', 'Conf'])
            writer.writeheader()
            writer.writerows(rows)

    def _save_image(self, path, image):
        cv2.imwrite(path, image)

    def save_csv(self, path, rows):
        self.queue.put({'type': 'csv', 'path': path, 'rows': rows})

    def save_image(self, path, image):
        self.queue.put({'type': 'image', 'path': path, 'image': image})

    def flush(self):
        self.queue.join()
