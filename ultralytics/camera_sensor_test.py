import threading
import asyncio
import threading
import time
import os
import cv2
import paho.mqtt.client as mqtt

from ultralytics.tracknet.engine.model import TrackNet
from ultralytics.tracknet.pred_stream_dataset import ImageBufferDataset
from ultralytics.tracknet.protocal.image_buffer import FakeFrame, FakeImageBuffer, ImageBufferProtocol

from ultralytics.yolo.utils import LOGGER

# STEP 1
# 起一個 thread 建立 mqtt server
# docker run -it --rm -p 1883:1883 eclipse-mosquitto

# STEP 2
# 起一個 thread 讀特定檔案，一張一張塞入 FakeImageBuffer
class ImageFeederThread(threading.Thread):
    def __init__(self, image_dir, buffer):
        super().__init__(daemon=True)
        self.image_dir = image_dir
        self.buffer = buffer

    def run(self):
        idx = 0 
        for filename in sorted(os.listdir(self.image_dir)):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(self.image_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    is_eos = False
                    if idx == 300:
                        is_eos = True
                    frame = FakeFrame(img, index=idx, is_eos=is_eos)
                    idx += 1
                    self.buffer.push(frame)
                    if is_eos:
                        print("[Feeder] Push EOS")
                        break
                    

# STEP 3
# 起一個 thread 執行 model.predict()，使用 ImageBufferDataset
class TrackNet1000Thread:
    def __init__(self, mqttc:mqtt.Client,
                 output_topic:str,
                 output_width:int, output_height:int,
                 model_path:str,
                 imgbuf: ImageBufferProtocol):

        self.images = []
        self.fids = []
        self.timestamps = []
        self.output_width = output_width
        self.output_height = output_height
        self.mqttc = mqttc
        self.output_topic = output_topic
        # wait for new image
        self.isProcessing = False

        overrides = {}
        overrides['model'] = model_path
        overrides['mode'] = 'predict_v2'
        overrides['data'] = 'tracknet.yaml'
        overrides['batch'] = 1
        self.model = TrackNet(overrides, mqttc=mqttc, output_topic=output_topic, dataset=ImageBufferDataset(imgbuf))

    def start(self):
        #logging.debug("TrackNetThread started.")
        self.model.predict()
        # try:
        #     self.model.predict()
        # except Exception as e:
        #     LOGGER.error(e)
        # finally:
        #     self.isProcessing = False

if __name__ == "__main__":

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] ✅ Connected to broker")
        else:
            print(f"[MQTT] ❌ Failed to connect. Code {rc}")

    def on_disconnect(client, userdata, rc):
        print(f"[MQTT] 🔌 Disconnected with code {rc}")

    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    try:
        mqtt_client.connect("localhost", 1883)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT] ❌ Exception during connect: {e}")

    # 建立共享 buffer
    image_buffer = FakeImageBuffer()

    # 啟動圖片餵入
    image_path = r'/Users/bartek/git/BartekTao/datasets/tracknet/train_data/match_2/frame/1_00_01/'
    image_path = r'/usr/src/datasets/tracknet/val_data/sport_ai_2048_1536/frame/CameraReader_1/'
    feeder = ImageFeederThread(image_path, image_buffer)
    feeder.start()

    time.sleep(1)

    # 啟動推論線程
    model_path = r'/Users/bartek/git/BartekTao/ultralytics/runs/detect/train178/weights/last.pt'
    model_path = r'/usr/src/ultralytics/runs/detect/train637/weights/best.pt'
    predictor = TrackNet1000Thread(mqtt_client, "predict/result", 640, 640, model_path, image_buffer)
    predictor.start()

    # 訂閱用戶端（可省略）
    def on_message(client, userdata, msg):
        print("[Client] Got:", msg.payload.decode())

    sub = mqtt.Client()
    sub.on_message = on_message
    sub.connect("localhost", 1883)
    sub.subscribe("predict/result")
    sub.loop_start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped")