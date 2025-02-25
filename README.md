# Facial-Detection
A facial detection software with a primary focus on VTuber applications.
<br/>
To use, run datasetMaker to create a dataset of desired length <br/>
run trainingset to mark the eyes and mouth (eye, eye, mouth) on each image in the set <br/>
run luminancePrecalculation to make the lookup table for all the luminance values in each image. This will take a while. <br/>
run facial_detection for the desired amount of iterations. Single to double digit thousands advised for best results. <br/>
run showMeWhatYou'veGot to see the network that has been generated analyze a random image from the set.
