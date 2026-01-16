import sys
import time
import threading
import queue
import cv2
import olympe
import numpy as np
from ultralytics import YOLO
from olympe.messages.ardrone3.MediaStreaming import VideoEnable
from olympe.messages.ardrone3.Piloting import TakeOff, Landing
from pynput import keyboard

# --- 配置參數 ---
DRONE_IP = "192.168.42.1"
MODEL_WTS = "/home/allwn/桌面/best_ikea_3.pt"

class DroneVision:
    def __init__(self):
        # 1. 初始化無人機，不使用外部 RTSP
        self.drone = olympe.Drone(DRONE_IP)
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1) # 設為 1 確保偵測不延遲
        
        # 2. 載入 YOLO (ASUS TUF 強制開啟 GPU)
        try:
            self.model = YOLO(MODEL_WTS)
            self.model.to('cuda')
            print("YOLOv8 (GPU 加速) 準備就緒")
        except Exception as e:
            print(f"GPU 啟動失敗，使用 CPU: {e}")
            self.model = YOLO(MODEL_WTS)

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try: k = key.char.lower()
        except AttributeError: k = key.name
        if k == 't': self.drone(TakeOff())
        elif k == 'l': self.drone(Landing())
        elif k == 'q': self.running = False; return False

    def yuv_frame_cb(self, frame):
        """ 核心回調：直接處理底層 YUV 數據，徹底避開 RTSP 404 錯誤 """
        if not self.running:
            return
        
        try:
            # 1. 檢查幀是否有效，並過濾 Metadata
            info = frame.info()
            if info is None or info["format"] == "metadata":
                return

            # 2. 直接從記憶體提取原始數據 (ndarray)
            # 這是不經過網路協議的最穩定方式
            yuv_ndarray = frame.as_ndarray()
            if yuv_ndarray is None:
                return

            # 3. 將 Anafi 的 YUV420 格式轉為 OpenCV 的 BGR
            # 這是手動解碼，穩定性最高
            cv2_frame = cv2.cvtColor(yuv_ndarray, cv2.COLOR_YUV2BGR_I420)

            # 4. 更新影像隊列
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(cv2_frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(cv2_frame)
                except:
                    pass
        except Exception:
            pass

    def start(self):
        print(f"正在連線至 Anafi (IP: {DRONE_IP})...")
        if not self.drone.connect():
            print("連線失敗，請檢查 WiFi")
            return

        # 啟動無人機影片廣播
        print("發送 VideoEnable 指令...")
        self.drone(VideoEnable(1)).wait()
        
        # 關鍵：使用 Olympe 內建的 Streaming 回調，這不是走 RTSP 端口 554
        # 而是走 Olympe 的內部通道
        self.drone.streaming.set_callbacks(raw_cb=self.yuv_frame_cb)
        self.drone.streaming.start()

        print("[Native Mode] 視窗啟動中，請等待同步...")

        cv2.namedWindow("Taipower AI Monitoring", cv2.WINDOW_NORMAL)

        try:
            while self.running:
                try:
                    # 從隊列拿圖
                    img = self.frame_queue.get(timeout=2.0)
                    
                    # 執行 YOLO 辨識
                    results = self.model.predict(img, conf=0.4, verbose=False, imgsz=416)
                    display_img = results[0].plot()
                    
                    # 加入狀態標籤
                    cv2.putText(display_img, "LIVE NATIVE STREAM", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except queue.Empty:
                    # 沒圖時顯示黑色背景
                    display_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_img, "SEARCHING NATIVE SIGNAL...", (130, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("Taipower AI Monitoring", display_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        except Exception as e:
            print(f"發生異常: {e}")
        finally:
            self.stop()

    def stop(self):
        print("\n安全關閉所有資源...")
        self.running = False
        try:
            self.drone.streaming.stop()
        except:
            pass
        self.drone.disconnect()
        cv2.destroyAllWindows()
        print("任務已安全結束")

if _name_ == "_main_":
    vision = DroneVision()
    vision.start()
撰寫內容給洪靖倫
