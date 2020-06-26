This repository features the code used and results obtained during my bachelor thesis for artificial intelligence. <br>
<br>
I sought to analyze the differences caused by expanding a standard evolutionary game by implementing asymmetry and social networks. <br>
Running `model.py` will start a simulation running four different versions of the evolutionary game model. This is done using the `run()` function to allow an average to be created of the different simulation. Currently the file will run the simulation 20 times with 100 generations. These can be altered by changing the parameter values in the `run()` function. Different parameters can be passed to change the execution of the simulation. A model can also be run individually by creating a Model object and using the simulate method.
<br>

The results of a default simulation can be found in the folders in this repository. They contain per model version the progress of the strategy distribution and average score. There is also a graph containing all average score progression to allow for easier comparison. The data.txt file contains a brief overview of the strategies per simulation (20 in total) containing per strategy: the number of students using it, the max score, the min score and difference.  
