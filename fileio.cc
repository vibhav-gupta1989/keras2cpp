#include "fileio.h"
#include<fstream>
#include<iostream>

using namespace std;

void read_file(const string &fname, int& m_rows, int& m_cols, int& m_depth, 
    vector<vector<vector<float>>>& data){

    char tmp_char;
    float tmp_float;
    ifstream fin(fname.c_str());
    fin  >> m_rows >> m_cols >> m_depth;

    for(int i=0; i<m_rows; i++){
        fin >> tmp_char;
        vector<vector<float>>c;
        for(int j=0; j<m_cols; j++){
            vector<float>d;
            for(int k=0; k<m_depth; k++){
                fin >> tmp_float;
                d.push_back(tmp_float);
            }
            c.push_back(d);
        }
        fin >> tmp_char;
        data.push_back(c);
    }
    fin.close();
}

void load_weights_conv2d(ifstream &fin, int& m_rows, int& m_cols, int& m_depth, 
        int& m_kernels_cnt, std::string& m_border_mode, 
        vector<vector<vector<vector<float>>>>& rows, vector<float>& m_bias){
    
    char tmp_char = ' ';
    string tmp_str = "";
    float tmp_float;
    bool skip = false;
    fin >> m_rows >> m_cols >> m_depth >> m_kernels_cnt >>  m_border_mode;
    if (m_border_mode == "[") { m_border_mode = "valid"; skip = true; }

    cout << "LayerConv2D " << m_kernels_cnt << "x" << m_depth << "x" << m_rows <<
              "x" << m_cols << " border_mode " << m_border_mode << endl;
    // reading kernel weights

    for(int i=0; i<m_rows; i++){
        vector<vector<vector<float>>> cols;
        for(int j=0; j<m_cols; j++){
            vector<vector<float>>depths;
            for(int d=0; d<m_depth; d++){
                fin >> tmp_char;
                vector<float>filters;
                for(int f=0; f<m_kernels_cnt; f++){
                    fin >> tmp_float;
                    filters.push_back(tmp_float);
                }
                fin >> tmp_char;
                depths.push_back(filters);
            }
            cols.push_back(depths);
        }
        rows.push_back(cols);
    }

    // reading kernel biases
    fin >> tmp_char; // for '['
    for(int k = 0; k < m_kernels_cnt; ++k) {
        fin >> tmp_float;
        cout << tmp_float << endl;
        m_bias.push_back(tmp_float);
    }
    fin >> tmp_char; // for ']'

}

void load_weights_activation(std::ifstream& fin, std::string& m_activation_type){
    fin >> m_activation_type;
    cout << "Activation type " << m_activation_type << endl;

}

void load_weights_max_pooling(std::ifstream& fin, int& m_pool_x, int& m_pool_y){
    fin >> m_pool_x >> m_pool_y;
    cout << "MaxPooling " << m_pool_x << "x" << m_pool_y << endl;
}

void load_weights_dense(std::ifstream& fin, int& m_input_cnt, int& m_neurons, 
        std::vector<std::vector<float>>& m_weights, std::vector<float>& m_bias){
    
    fin >> m_input_cnt >> m_neurons;
    float tmp_float;
    char tmp_char = ' ';
    for(int i = 0; i < m_input_cnt; ++i) {
        vector<float> tmp_n;
        fin >> tmp_char; // for '['
        for(int n = 0; n < m_neurons; ++n) {
            fin >> tmp_float;
            tmp_n.push_back(tmp_float);
        }
        fin >> tmp_char; // for ']'
        m_weights.push_back(tmp_n);
    }
    cout << "weights " << m_weights.size() << endl;
    fin >> tmp_char; // for '['
    for(int n = 0; n < m_neurons; ++n) {
        fin >> tmp_float;
        m_bias.push_back(tmp_float);
    }
    fin >> tmp_char; // for ']'
    cout << "bias " << m_bias.size() << endl;

}

void get_layers_cnt(std::ifstream&fin, std::string& tmp_str, int& m_layers_cnt){
    fin >> tmp_str >> m_layers_cnt;
}

void get_layer_type(std::ifstream& fin, std::string& tmp_str, int& tmp_int, std::string& layer_type){
    fin >> tmp_str >> tmp_int >> layer_type;
}