#include "layers.h"
#include "fileio.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

void layers::DataChunk2D::read_from_file(const std::string &fname) {
  read_file(fname, m_rows, m_cols, m_depth, data);
}


void layers::LayerConv2D::load_weights(std::ifstream &fin) {
  load_weights_conv2d(fin, m_rows, m_cols, m_depth, m_kernels_cnt, m_border_mode, rows, m_bias);
}

void layers::LayerActivation::load_weights(std::ifstream &fin) {
  load_weights_activation(fin, m_activation_type);
}

void layers::LayerMaxPooling::load_weights(std::ifstream &fin) {
  load_weights_max_pooling(fin, m_pool_x, m_pool_y);
}

void layers::LayerDense::load_weights(std::ifstream &fin) {
  load_weights_dense(fin, m_input_cnt, m_neurons, m_weights, m_bias);
}

layers::KerasModel::KerasModel(const string &input_fname, bool verbose)
                                                       : m_verbose(verbose) {
  load_weights(input_fname);
}


layers::DataChunk* layers::LayerFlatten::compute_output(layers::DataChunk* dc) {

  try{
    vector<vector<vector<float> > > im = dc->get_3d();

    size_t csize = im[0].size();
    size_t rsize = im[0][0].size();
    size_t size = im.size() * csize * rsize;
    layers::DataChunkFlat *out = new DataChunkFlat(size);
    float * y_ret = out->get_1d_rw().data();
    for(size_t i = 0, dst = 0; i < im.size(); ++i) {
      for(size_t j = 0; j < csize; ++j) {
        float * row = im[i][j].data();
        for(size_t k = 0; k < rsize; ++k) {
          y_ret[dst++] = row[k];
        }
      }
    }

    return out;
  }
  catch(...){
    return NULL;
  }
}


layers::DataChunk* layers::LayerMaxPooling::compute_output(layers::DataChunk* dc) {
  try{
    vector<vector<vector<float> > > im = dc->get_3d();
    vector<vector<vector<float> > > y_ret;

    for(int i=0; i<im.size()/m_pool_x; i++){
      vector<vector<float>>c;
      for(int j=0; j<im[0].size()/m_pool_y; j++){
        vector<float>f;
        for(int k=0; k<im[0][0].size(); k++){
          f.push_back(0);
        }
        c.push_back(f);
      }
      y_ret.push_back(c);
    }
  
    for(unsigned int d = 0; d < y_ret[0][0].size(); ++d) {
      for(unsigned int x = 0; x < y_ret.size(); ++x) {
        unsigned int start_x = x*m_pool_x;
        unsigned int end_x = start_x + m_pool_x;
        for(unsigned int y = 0; y < y_ret[0].size(); ++y) {
          unsigned int start_y = y*m_pool_y;
          unsigned int end_y = start_y + m_pool_y;

          vector<float> values;
          for(unsigned int i = start_x; i < end_x; ++i) {
            for(unsigned int j = start_y; j < end_y; ++j) {
              values.push_back(im[i][j][d]);
            }
          }
          y_ret[x][y][d] = *max_element(values.begin(), values.end());
        }
      }
    }
    layers::DataChunk *out = new layers::DataChunk2D();
    out->set_data(y_ret);
    return out;
  }
  catch(...){
    return NULL;
  }
}

void layers::missing_activation_impl(const string &act) {
  std::cout << "Activation " << act << " not defined!" << std::endl;
  std::cout << "Please add its implementation before use." << std::endl;
  //exit(1);
}

layers::DataChunk* layers::LayerActivation::compute_output(layers::DataChunk* dc) {

  try{
    if (dc->get_data_dim() == 3) {
      vector<vector<vector<float> > > y = dc->get_3d();
      if(m_activation_type == "relu") {
        for(unsigned int i = 0; i < y.size(); ++i) {
          for(unsigned int j = 0; j < y[0].size(); ++j) {
            for(unsigned int k = 0; k < y[0][0].size(); ++k) {
              if(y[i][j][k] < 0) y[i][j][k] = 0;
            }
          }
        }
        layers::DataChunk *out = new layers::DataChunk2D();
        out->set_data(y);
        return out;
      } else {
        layers::missing_activation_impl(m_activation_type);
        throw("missing activation implementation");
      }
    } else if (dc->get_data_dim() == 1) { // flat data, use 1D
      vector<float> y = dc->get_1d();
      if(m_activation_type == "relu") {
        for(unsigned int k = 0; k < y.size(); ++k) {
          if(y[k] < 0) y[k] = 0;
        }
      } else if(m_activation_type == "softmax") {
        float sum = 0.0;
        for(unsigned int k = 0; k < y.size(); ++k) {
          y[k] = exp(y[k]);
          sum += y[k];
        }
        for(unsigned int k = 0; k < y.size(); ++k) {
          y[k] /= sum;
        }
      } else if(m_activation_type == "sigmoid") {
        for(unsigned int k = 0; k < y.size(); ++k) {
          y[k] = 1/(1+exp(-y[k]));
        }
      } else if(m_activation_type == "tanh") {
        for(unsigned int k = 0; k < y.size(); ++k) {
          y[k] = tanh(y[k]);
        }
      } else {
        layers::missing_activation_impl(m_activation_type);
        throw("missing activation implementation");
      }

      layers::DataChunk *out = new DataChunkFlat();
      out->set_data(y);
      return out;
    } else { throw "data dim not supported"; }

    return dc;
  }
  catch(...){
    return NULL;
  }
}

layers::DataChunk* layers::LayerConv2D::compute_output(layers::DataChunk* dc) {

  try{
    auto const & im = dc->get_3d();
    vector<vector<vector<float>>> y_ret;

    int rowOffset = 0;
    int colOffset = 0;
    if(m_border_mode == "valid"){
      rowOffset = m_rows/2;
      colOffset = m_cols/2;
    }
  
    for(int i=rowOffset; i<im.size()-rowOffset; i++){
      vector<vector<float>>c;
      for(int j=colOffset; j<im[0].size()-colOffset; j++){
        vector<float>f;
        for(int k=0; k<m_kernels_cnt; k++){
          f.push_back(0);
        }
        c.push_back(f);
      }
      y_ret.push_back(c);
    }

    float output;

    for(int i=0; i<y_ret.size(); i++){
      for(int j=0; j<y_ret[0].size(); j++){
        for(int k=0; k<m_kernels_cnt; k++){
          output = 0;
          for(int dd=0; dd < m_depth; dd++){

            if(m_border_mode == "same"){
              for(int l=i-(m_rows/2), n=0; l<=i+(m_rows/2); l++, n++){
                for(int m=j-(m_cols/2), o=0; m<=j+(m_cols/2); m++, o++){
                  if(l < 0 || l >= im.size() || m < 0 || m >= im[0].size())
                    continue;

                  output += im[l][m][dd] * rows[n][o][dd][k];
                } 
              }
            }

            else{
              int row_id, col_id;
              row_id = i + (m_rows/2);
              col_id = j + (m_cols/2);

              for(int l=row_id-(m_rows/2), n=0; l<=row_id+(m_rows/2); l++, n++){
                for(int m=col_id-(m_cols/2), o=0; m<=col_id+(m_cols/2); m++, o++){
                  output += im[l][m][dd] * rows[n][o][dd][k]; 
                }
              }
            }
          }
          y_ret[i][j][k] = output + m_bias[k];
        }
      }
    }


    layers::DataChunk *out = new layers::DataChunk2D();
    out->set_data(y_ret);
    return out;
  }
  catch(...){
    return NULL;
  }
}

layers::DataChunk* layers::LayerDense::compute_output(layers::DataChunk* dc) {
  //cout << "weights: input size " << m_weights.size() << endl;
  //cout << "weights: neurons size " << m_weights[0].size() << endl;
  //cout << "bias " << m_bias.size() << endl;
  try{
    size_t size = m_weights[0].size();
    size_t size8 = size >> 3;
    layers::DataChunkFlat *out = new DataChunkFlat(size, 0);
    float * y_ret = out->get_1d_rw().data();

    auto const & im = dc->get_1d();

    for (size_t j = 0; j < m_weights.size(); ++j) { // iter over input
      const float * w = m_weights[j].data();
      float p = im[j];
      size_t k = 0;
      for (size_t i = 0; i < size8; ++i) { // iter over neurons
        y_ret[k]   += w[k]   * p;          // vectorize if you can
        y_ret[k+1] += w[k+1] * p;
        y_ret[k+2] += w[k+2] * p;
        y_ret[k+3] += w[k+3] * p;
        y_ret[k+4] += w[k+4] * p;
        y_ret[k+5] += w[k+5] * p;
        y_ret[k+6] += w[k+6] * p;
        y_ret[k+7] += w[k+7] * p;
        k += 8;
      }
      while (k < size) { y_ret[k] += w[k] * p; ++k; }
    }
    for (size_t i = 0; i < size; ++i) { // add biases
      y_ret[i] += m_bias[i];
    }

    return out;
  }
  catch(...){
    return NULL;
  }
}


std::vector<float> layers::KerasModel::compute_output(layers::DataChunk *dc) {
  //cout << endl << "KerasModel compute output" << endl;
  //cout << "Input data size:" << endl;
//  dc->show_name();

  layers::DataChunk *inp = dc;
  layers::DataChunk *out = 0;
  for(int l = 0; l < (int)m_layers.size(); ++l) {
    std::cout << "Processing layer " << m_layers[l]->get_name() << std::endl;
    
    auto start = high_resolution_clock::now();
    out = m_layers[l]->compute_output(inp);
    if(out == NULL){
      std::cout << m_layers[l]->get_name() << " layer failed" << std::endl;
      exit(1);
    }
    else{
      std::cout << m_layers[l]->get_name() << " layer succeeded" << std::endl;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    std::cout << "Time taken is " << duration.count() << " microseconds" << std::endl; 

    //cout << "Input" << endl;
    //inp->show_name();
    //cout << "Output" << endl;
    //out->show_name();
    if(inp != dc) delete inp;
    //delete inp;
    inp = 0L;
    inp = out;
  }

  std::vector<float> flat_out = out->get_1d();
  std::cout << std::endl << "Prediction Output:" << std::endl;
  out->show_values();
  delete out;

  return flat_out;
}

void layers::KerasModel::load_weights(const string &input_fname) {
  if(m_verbose) std::cout << "Reading model from " << input_fname << std::endl;
  ifstream fin(input_fname.c_str());
  string layer_type = "";
  string tmp_str = "";
  int tmp_int = 0;

  get_layers_cnt(fin, tmp_str, m_layers_cnt);
  if(m_verbose) std::cout << "Layers " << m_layers_cnt << std::endl;

  for(int layer = 0; layer < m_layers_cnt; ++layer) { // iterate over layers
    get_layer_type(fin, tmp_str, tmp_int, layer_type);
    if(m_verbose) std::cout << tmp_str << " " << "Layer " << tmp_int << " " << layer_type << std::endl;

    Layer *l = 0L;
    if(layer_type == "Convolution2D") {
      l = new LayerConv2D();
    } else if(layer_type == "Activation") {
      l = new LayerActivation();
    } else if(layer_type == "MaxPooling2D") {
      l = new LayerMaxPooling();
    } else if(layer_type == "Flatten") {
      l = new LayerFlatten();
    } else if(layer_type == "Dense") {
      l = new LayerDense();
    } else if(layer_type == "Dropout") {
      continue; // we dont need dropout layer in prediciton mode
    }
    if(l == 0L) {
      std::cout << "Layer is empty, maybe it is not defined? Cannot define network." << std::endl;
      return;
    }
    l->load_weights(fin);
    m_layers.push_back(l);
  }

  fin.close();
}

layers::KerasModel::~KerasModel() {
  for(int i = 0; i < (int)m_layers.size(); ++i) {
    delete m_layers[i];
  }
}

int layers::KerasModel::get_output_length() const
{
  int i = m_layers.size() - 1;
  while ((i > 0) && (m_layers[i]->get_output_units() == 0)) --i;
  return m_layers[i]->get_output_units();
}
