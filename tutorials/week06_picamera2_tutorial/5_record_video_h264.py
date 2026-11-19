# 5_record_video_h264.py
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time

picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": (1920, 1080), "format": "YUV420"},
                                        controls={"FrameDurationLimits": (33333, 33333)})  # ~30 fps
picam2.configure(cfg)
picam2.start()

time.sleep(0.8)  # let AE/AWB settle, then lock for consistent colour
picam2.set_controls({"AeEnable": False, "AwbEnable": False})

encoder = H264Encoder(bitrate=8_000_000)
output  = FfmpegOutput("clip.mp4")   # wraps H.264 into MP4 container
picam2.start_recording(encoder, output)
print("Recording 5 s of 1080p30…")
time.sleep(5)
picam2.stop_recording()
picam2.stop()
print("Saved clip.mp4")
