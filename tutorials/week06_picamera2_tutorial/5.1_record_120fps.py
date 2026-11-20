# 5.1_record_120fps.py
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput
import time

# Size of the recorded video. The Raspberry Pi 5 encodes the video with its processor,
# and at the full 1332x990 it cannot keep up with 120 fps (a 5 s clip then has only
# about 520 frames instead of 600). 888x660 is 2/3 of the sensor image in each
# direction, with the same shape and the same field of view, and it keeps all frames.
# If your clip still has clearly fewer than 600 frames, make it smaller, for example
# (592, 440). Keep the height a multiple of 4. With other heights, the colours in the
# video are shifted. The sensor runs at 120 fps whatever size you choose.
video_size = (888, 660)

picam2 = Picamera2()
# The same sensor setting as script 3.1: the 1332x990 sensor mode with 10 bits per pixel,
# and a frame time of 8333 microseconds (about 1/120 of a second).
cfg = picam2.create_video_configuration(main={"size": video_size, "format": "YUV420"},
                                        sensor={"output_size": (1332, 990), "bit_depth": 10},
                                        controls={"FrameDurationLimits": (8333, 8333)})
picam2.configure(cfg)
# It should print {'bit_depth': 10, 'output_size': (1332, 990)}.
print("Sensor mode in use:", picam2.camera_configuration()["sensor"])
picam2.start()

# Let the automatic exposure and white balance settle for 1 s, then turn them off,
# so that the brightness and colour stay the same during the recording.
time.sleep(1)
picam2.set_controls({"AeEnable": False, "AwbEnable": False})

# The Raspberry Pi 5 and the CM5 have no hardware video encoder, so the processor (CPU)
# encodes the video.
# framerate=120 tells the encoder that the video has 120 frames per second (the default is 30).
encoder = H264Encoder(bitrate=10_000_000, framerate=120)
# PyavOutput writes an MP4 file in which every frame keeps the camera's own timestamp.
output = PyavOutput("clip_120fps.mp4")
picam2.start_recording(encoder, output)
print("Recording 5 s at 120 fps...")
time.sleep(5)
picam2.stop_recording()
picam2.stop()
print("Saved clip_120fps.mp4")
print("A 5-second clip at 120 fps should have about 600 frames. To count them, run:")
print("ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 clip_120fps.mp4")
