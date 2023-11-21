from threading import Thread
import time
import cv2
        

class GetCap():
    def __init__(self, src=0):
        self.stopped = False
        self.path = src
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
        self.stream.set(3, 1920)
        self.stream.set(4, 1080)
        assert self.stream.isOpened(), 'Cannot capture source'
        (self.cap, self.frame) = self.stream.read()
        self.count = 1

    def start(self):
        t = Thread(target=self.get_video, args=(), daemon=True)
        t.start()
        return self

    def get_video(self):
        while True:
            if self.stopped:
                return
            if not self.cap:
                print('not cap')
                self.stop()
            else: 
                (self.cap, self.frame) = self.stream.read()
                self.count += 1

    def stop(self):
        self.stopped = True
        self.stream.release()
        time.sleep(0.2)
