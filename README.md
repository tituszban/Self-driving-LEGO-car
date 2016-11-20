# Self-driving-LEGO-car

I have built a remote controlled LEGO car, with double engines using an adder, differential drive and four wheel independed suspension. The controll of the robot was done through Bluetooth with a LEGO EV3 controller brick, using a C# interface, that converted a simple steer-throttle input to direct motor instructions.

The car was tested and trained using a user interface, made in Unity3D, that took user inputs, and sent it through the interface. Also, the car was equiped with a tablet(Samsung Tab 4), used as a camera with an IP webcam app (https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_GB). From the webcam server a logger developed in Python 2.7 took the images 5 times a second, and recorded the current controll inputs. The images then were compressed (to 40x40 pixels) and modified (greyscale, gaussian blur, edge detection).

Using the database built from the images two neuro-network was trained in Octave, for both steering and the throttle. Using this another python program then took the picture from the server, processed it and put it through the neuro-networks, determining the inputs, then transmitted it through the interface.

For demostration on driving and image processing see: https://youtu.be/kZt32APp5ug

One of the interesting parts of this project was the simultaneous using of three different programming languages. The car interface took the inputs from a file, that C# programs can edit without probelms, using mutexes, but not Pyton programs. So in order to keep the integrity of the files, the Python logger and neuro-network controller was given a .NET library, also made by me, to be able to use mutexes. I also had to use a Python framework in order to run the neuro-network with the real time values in octave. I managed all these connection with relativly low latency, as the bottleneck in update speed was the bluetooth communication.
