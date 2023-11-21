from queue import Queue
from threading import Thread
import json
import time
from socket import *
from collections import defaultdict
import cv2
import os
from urllib.parse import urljoin
import requests
import asyncio
import websockets


class StartTCP():
    def __init__(self, ip_add, port):
        self.tcp_server = socket(AF_INET, SOCK_STREAM)
        self.address = (ip_add, port)
        self.ip_add = ip_add
        self.port = port
    def start(self):
        self.tcp_server.bind(self.address)
        self.tcp_server.listen(128)
        print('-------------- Start listening to {} --------------'.format(self.port))
        while True:
            client, addr = self.tcp_server.accept()
            return client
        

class SendMsg():
    def __init__(self, client, queueSize=1024):
        self.Q = Queue(maxsize=queueSize)
        self.stoped = False
        self.client = client

    def start(self):
        t = Thread(target=self.send_msg, args=(), daemon=True)
        t.start()
        return self

    def send_msg(self):
        while True:
            if self.stoped:
                return
            if not self.Q.empty():
                keypoints = self.Q.get()
                if keypoints is not None:
                    # print(keypoints)
                    msg = json.dumps(keypoints)
                    self.client.send(msg.encode('utf-8'))
            else:
                time.sleep(0.1)
    
    def save(self, kpts):
        self.Q.put(kpts)

    def stop(self, stream):
        self.stoped = True
        for i in range(len(stream)):
            stream[i].release()
        self.client.close()
        time.sleep(0.2)


class SendVideo():
    def __init__(self, path):
        self.path = path
        self.frames_to_save = defaultdict(list)
        self.flag = -1

    def start(self):
        t = Thread(target=self.connect, args=(), daemon=True)
        t.start()
        return self
    
    async def websocket_client(self):
        # uri = "ws://127.0.0.1:8000/camListener"
        uri = "ws://10.11.140.36:8000/camListener"
        async with websockets.connect(uri) as websocket:
            async for message in websocket:
                print(message)
                if 'start' in message: # start recording video
                    self.flag = 0
                    print('start')
                if 'stop' in message: # stop recording video and send the video path to server
                    self.flag = 1
                    print('stop')
                if 'reset' in message:
                    self.flag = 2
                    print('reset')
                if 'action' in message:
                    self.flag = 3
                    print('action')

    def connect(self):
        asyncio.run(self.websocket_client())

    def send_video(self):
        while True:
            if self.frames_to_save is not None:
                for l_i in self.frames_to_save:
                    size = (1920, 1080)
                    fps = 30
                    out_fn = os.path.join(self.path, f"{str(l_i)}.avi")
                    result = cv2.VideoWriter(out_fn,
                                            cv2.VideoWriter_fourcc(*'DIVX'),
                                            fps, size)
                    for i, frame in enumerate(self.frames_to_save[l_i]):
                        result.write(frame)
                    result.release()
                url = urljoin('http://10.11.140.36:8000/', 'videoLocation/')
                video_path = os.path.join(self.path, "0.avi")
                data = {'video': video_path}
                ret = requests.post(url, json=data)
                self.flag = -1
                break
            else:
                print("=====no data in video!=====")
                time.sleep(2)

    
    def save(self, frames_to_save):
        self.frames_to_save = frames_to_save
        self.send_video()
