Original Repo: https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master

python src/detector.py --resolution 640x480

python src/detector.py --resolution 640x480 --debug

ffmpeg -video_size 640x480 -i /dev/video0 -input_format mjpeg -f mpegts "udp://192.168.1.14:5000"
