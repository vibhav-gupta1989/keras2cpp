import numpy as np
np.random.seed(1337)
from keras.models import Sequential, model_from_json, load_model
import json
import argparse

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description='This is a simple script to dump Keras model into simple format suitable for porting into pure C++ model')

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-o', '--output', help="Ouput file name", required=True)
parser.add_argument('-v', '--verbose', help="Verbose", required=False)
#args = parser.parse_args()

architecture = "example/my_nn_arch.json";
weights = "example/my_nn_weights.h5";
output = "example/dumped.nnet";
verbose = True;

print('Read architecture from', architecture)
print('Read weights from', weights)
print('Writing to', output)

arch = open(architecture).read()
# model = model_from_json(arch)
# modelJsonString = model.to_json()
# model.load_weights(args.weights)
# model.compile(loss='categorical_crossentropy', optimizer='adadelta')
arch = json.loads(arch)
model = load_model("example/my_model.keras")

with open(output, 'w') as fout:
    fout.write('layers ' + str(len(model.layers)) + '\n')

    layers = []
    for ind, l in enumerate(model.layers):
        if verbose:
            print(ind, l)
        # fout.write('layer ' + str(ind) + ' ' + l.name + '\n')

        if verbose:
            print(str(ind), l.name)
        layers += [l.name]
        if 'conv2d' in l.name:
            fout.write('layer ' + str(ind) + ' ' + 'Convolution2D' + '\n')
            #fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')

            #if 'batch_input_shape' in l['config']:
            #    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
            #fout.write('\n')

            W = l.get_weights()[0]
            if verbose:
                print(W.shape)
            
            #W = W.reshape(W.shape[3], W.shape[2], W.shape[0], W.shape[1])
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l.padding + '\n')
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    for k in range(W.shape[2]):
                        fout.write(str(W[i,j,k]) + '\n')
            fout.write(str(l.get_weights()[1]) + '\n')

        if 'activation' in l.name:
            # fout.write(l['config']['activation'] + '\n')
            fout.write('layer ' + str(ind) + ' ' + 'Activation' + '\n')
            if(l.activation._api_export_path =='keras.activations.relu'):
                print(l.activation)
                fout.write('relu\n')
            else:
                fout.write('softmax')
        if 'max_pooling' in l.name:
            fout.write('layer ' + str(ind) + ' ' + 'MaxPooling2D' + '\n')
            #fout.write("2 2\n")
            fout.write(str(l.pool_size[0]) + ' ' + str(l.pool_size[1]) + '\n')
        #if l['class_name'] == 'Flatten':
        #    print l['config']['name']
        if 'dense' in l.name:
            fout.write('layer ' + str(ind) + ' ' + 'Dense' + '\n')
            #fout.write(str(l['config']['output_dim']) + '\n')
            W = l.get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
            for w in W:
                fout.write(str(w) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')

        if 'dropout' in l.name:
            fout.write('layer ' + str(ind) + ' ' + 'Dropout' + '\n')

        if 'flatten' in l.name:
            fout.write('layer ' + str(ind) + ' ' + 'Flatten' + '\n')

