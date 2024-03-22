#include "layers.h"

using namespace std;
using namespace layers;

int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\n"
           << "Keras model will be used in C++ for prediction only." << endl;

  DataChunk *sample = new DataChunk2D();
  sample->read_from_file("./example/sample_mnist.dat");
  std::cout << sample->get_3d().size() << std::endl;
  KerasModel m("./example/dumped_mnist.nnet", true);
  m.compute_output(sample);
  delete sample;

  return 0;
}
