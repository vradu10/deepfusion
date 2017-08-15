# DeepFusion

Requires: Torch 7

DeepFusion is a framework designed to facilitate training and testing of deep neural network architectures for mobile and ubiquitous sensing.

This framework has been employed for analysis on four very different datasets, performing detections on tasks like: activity recognition, gait characterization, sleep stage detection and indoor-outdoor detection.

DeepFusion has the following features:

- is able to perform accurate detection directly on raw data, without any hand-crafted features required.
- permits transformation of sensor data to other feature domains: frequency domain and ecdf features extraction
- easy customize network training from one single file.

For most dataset, the default configuration file will perform In most condition training can be performed with the default variables provided in the configuration files, though occasionally it may be advised to change different configurations of these components.

To run the training simply run the following command

<pre>th main.lua</pre>

This will execute with the parameters specified in the configuration file and called in the ```main.lua``` file.

To try different parameters to train with, the code can be run with command line paramteres to perform different optimisations. For instance, changing the learning rate:

<pre> th main.lua -lr 0.01 </pre>

There are many parameters that can be used to override the parameters specified in the configuration file. A full list of these is presented below.


 Training can be performed with the following variables parameters.

|Options | Description [and default value] |
|--------|------------|  
|-batch_size|       mini-batch size [4]|
|  -net      |        network architecture index [1]|
|  -evaluation |      type of evaluation: cross-validation or leave-one-out [leave-one-out]|
|  -dataPreprocess |  Data preprocessing: time or freq or ecdf [time]|
|  -startingFold    | folds to start with [1]|
|  -endingFold      | folds to end with [10]|
|  -lr             |  learning rate [0.003]|
|  -momentum        | momentum [0.01]|
|  -dropout         | dropout probability [0.4]|
|  -epochs          | no of supervised learning epochs [5000]|
|  -resume          | resume training from file []|
|  -model           | continue training initial model []|
|  -gpuid           | which GPU to use; -1 is CPU [-1]|
|  -seed            | set seed for random number generation [123]|
|  -dataset_fraction |fraction of data used in training and test from original size in files [1]|
|  -verbose       |   verbose level: 0 - evaluation; 1 - debug [0]|
| -optim          |  optimization method: sgd | adam | sgdm  [sgd]|


Default values are overridden by configuration file and commandline parameters override configuration files.

## Providing your own data

Sensor data will be provided in a text file, following a specific data format.

```ax1, ax2, ..., ay1, ay2, ..., az1, az2, ..., gx1, gx2, ..., class```

, where axi is a stream of sensors data in one dimension and class is the relevant classification class.

There are several configuration modes:
- one file for both training and test (cross-validation), where the location of just one file needs to be included in the config file
- two files with training performed on one and test on the other. Again, their location needs to be specified in the config file.
- multiple file when evaluating with leave-one-user-out, files following the protocol of ```with_x.arff0``` for samples of subject x and ```without_x.arff0```, for samples of all the other subjects but x.
