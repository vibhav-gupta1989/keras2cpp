#ifndef FILEIO__H
#define FILEIO__H

#include<string>
#include<vector>

void read_file(const std::string &fname, int& m_rows, int& m_cols, int& m_depth, 
        std::vector<std::vector<std::vector<float>>>& data);

void load_weights_conv2d(std::ifstream &fin, int& m_rows, int& m_cols, int& m_depth, 
        int& m_kernels_cnt, std::string& m_border_mode, 
        std::vector<std::vector<std::vector<std::vector<float>>>>& rows, std::vector<float>& m_bias);

void load_weights_activation(std::ifstream& fin, std::string& m_activation_type);

void load_weights_max_pooling(std::ifstream& fin, int& m_pool_x, int& m_pool_y);

void load_weights_dense(std::ifstream& fin, int& m_input_cnt, int& m_neurons, 
        std::vector<std::vector<float>>& m_weights, std::vector<float>& m_bias);

void get_layers_cnt(std::ifstream& fin, std::string& tmp_str, int& m_layers_cnt);

void get_layer_type(std::ifstream& fin, std::string& tmp_str, int& tmp_int, std::string& layer_type);


#endif