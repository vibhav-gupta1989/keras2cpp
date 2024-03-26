# Run Instructions

### Step 1: 
Create and train a model in Python on the mnist dataset, save a sample image and run the prediction on the sample image.
```
cd example
python mnist_cnn_one_iteration.py
```

OR


Create and train a model in Python on the CIFAR 10 dataset, save a sample image and run the prediction on the sample image.
```
cd example
python cifar-10.py
```

### Step 2: 
Dump the model to a text file. By default, the script uses model and dumps it for mnist dataset. To use and dump model for the cifar-10 dataset, set in the script output to "example/dumped_cifar10.nnet" and argument for load_model to example/my_model_cifar10.keras. 
```
python dump_to_simple_cpp.py
```

### Step 3:
Compile C++ static libs for layers and fileio
```
g++ -c fileio.cc -o fileio.o
ar rcs libfileio.a fileio.o
g++ -c layers.cc -o layers.o
ar rcs liblayers.a layers.o
```

### Step 4: 
Compile the C++ code that has the test code implementation. By default, the code uses sample and model for the mnist dataset. To use the sample and model for cifar-10 dataset, specify the right location to the sample file and model in example_main.cc.
```
g++ -o example_main example_main.cc -L. -llayers -lfileio
```

### Step 5: 
Run the executable generated from Step 4. It loads the model and runs the test on the sample image. The output should match the prediction from Step 1.
```
./example_main
```
The output looks like below:-
```
This is simple example with Keras neural network model loading into C++.
Keras model will be used in C++ for prediction only.
Processing layer Conv2D
Conv2D layer succeeded
Time taken is 3684 microseconds
Processing layer Activation
Activation layer succeeded
Time taken is 2294 microseconds
Processing layer Conv2D
Conv2D layer succeeded
Time taken is 6663 microseconds
Processing layer Activation
Activation layer succeeded
Time taken is 1836 microseconds
Processing layer MaxPooling2D
MaxPooling2D layer succeeded
Time taken is 1151 microseconds
Processing layer Flatten
Flatten layer succeeded
Time taken is 158 microseconds
Processing layer Dense
Dense layer succeeded
Time taken is 380 microseconds
Processing layer Activation
Activation layer succeeded
Time taken is 123 microseconds
Processing layer Dense
Dense layer succeeded
Time taken is 240 microseconds
Processing layer Activation
Activation layer succeeded
Time taken is 600 microseconds

Prediction Output:
DataChunkFlat values:
0.0826641 0.0772677 0.0815491 0.0955249 0.0985643 0.119052 0.109999 0.124131 0.108191 0.103057
```