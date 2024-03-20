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
Dump the model to a text file
```
python dump_to_simple_cpp.py
```

### Step 3: 
Compile the C++ code that has the layers and the test code implementation
```
g++ example_main.cc -o output
```

### Step 4: 
Run the executable generated from Step 3. It loads the model and runs the test on the sample image. The output should match the prediction from Step 1.
```
./output
```
