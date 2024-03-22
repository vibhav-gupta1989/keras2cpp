# Run Instructions

### Step 1: 
Create and train a model in Python on the mnist dataset, save a sample image and run the prediction on a sample image.
```
cd example
python mnist_cnn_one_iteration.py
```

OR


Create and train a model in Python on the CIFAR 10 dataset, save a sample image and run the prediction on a sample image.
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