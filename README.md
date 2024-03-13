# Run Instructions

### Step 1: 
Create and train a model in Python on the mnist dataset and run the prediction on a sample image.
```
cd example
python mnist_cnn_one_iteration.py
```

### Step 2: 
Dump the model to a text file
```
python dump_to_simple_cpp.py
```

### Step 3: 
Load the model and sample image and implement the layers in C++ code
```
g++ example_main.cc -o output
```

### Step 4: 
Run the executable generated from Step 3. It runs the test on the sample image. The output should match the prediction from Step 1.
```
./output
```
