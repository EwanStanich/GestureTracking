Using:
- Python 3.10.14
- Tensorflow 2.16.1
- Opencv-python 4.9.0.80
- Mediapipe 0.10.11
- Other minor packages will either install automatically or errors will prompt to install them

To capture data, comment out the capture() function in gesturetrack.py and uncomment train_data(). The camera will turn on and prompt you on when data is being taken, with a brief break in between. 30 sets of 30 frames are taken, to capture the entire gesture and not just a single image. To add or change gestures, edit the action variable appropriately in both gesturetrack.py and train_data.py.

To train data, run train_data.py and the model will be output as action.h5.

To run prediction in real-time, uncomment the predictions() function in gesturetrack.py and run the file.
