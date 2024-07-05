#include <fstream>
#include <iostream>

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "phantom.h"
#include "softmax.cuh"
#include "utils.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

size_t N = 1ULL << 16;
size_t MM_LOG_N = 13;
size_t MM_N = 1ULL << MM_LOG_N;

double SCALE = pow(2.0, 40);

string TEST_TARGET = "MatMul";
vector<int> COEFF_MODULI =
    {60, 40, 60}  // MatMul
// {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}         // Softmax
// {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58} // LayerNorm
// {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58} // GELU
// {58, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
//  50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 55}             // 1763-bit
;

void MM_test() {
  EncryptionParameters parms(scheme_type::ckks);

  parms.set_poly_modulus_degree(MM_N);
  parms.set_coeff_modulus(CoeffModulus::Create(MM_N, COEFF_MODULI));

  PhantomContext context(parms);

  PhantomSecretKey secret_key(context);
  PhantomPublicKey public_key = secret_key.gen_publickey(context);
  PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

  std::vector<uint32_t> rots;
  for (int i = 0; i < MM_LOG_N; i++) {
    rots.push_back((MM_N + exponentiate_uint(2, i)) / exponentiate_uint(2, i));
  }
  PhantomGaloisKey galois_keys = secret_key.create_galois_keys_from_elts(context, rots);

  PhantomCKKSEncoder encoder(context);
  CKKSEvaluator ckks_evaluator(context, public_key, secret_key, encoder, relin_keys, galois_keys, SCALE);
  MMEvaluator mme(ckks_evaluator);

  // std::vector<std::vector<double>> matrix_4096x768 = mme.read_matrix("../../data/input/matrixmul_input_m_128_n_768_k_64_batch_128.txt", 4096, 768);
  // std::vector<std::vector<double>> matrix_768x64 = mme.read_matrix("../../data/input/matrix_input_n_768_k_64.txt", 768, 64);

  // vector<PhantomCiphertext> res;

  // auto matrix_4096x768_T = mme.transpose_matrix(matrix_4096x768);
  // auto matrix_768x64_T = mme.transpose_matrix(matrix_768x64);

  // std::vector<std::vector<double>> row_pack;

  // std::vector<double> row_ct(4096, 0.0);
  // for (auto i = 0; i < 64 * 768; i++) {
  //   int row = i / 768;
  //   int col = i % 768;
  //   row_ct[i % 4096] = matrix_768x64_T[row][col];
  //   if (i % 4096 == 4095) {
  //     row_pack.push_back(row_ct);
  //   }
  // }

  // mme.matrix_mul(matrix_4096x768_T, row_pack, res);

  // std::vector<std::vector<double>> matrix_4096x64 = mme.read_matrix("../../data/calibration/matrix_output_m_128_k_64_batch_128.txt", 4096, 64);
  // auto matrix_4096x64_T = mme.transpose_matrix(matrix_4096x64);

  // double average_err = 0.0;

  // // err of the first col
  // PhantomPlaintext res_pt;
  // vector<double> mm_res;
  // ckks_evaluator.decryptor.decrypt(res[0], res_pt);
  // ckks_evaluator.encoder.decode(res_pt, mm_res);
  // for (auto i = 0; i < 4096; i++) {
  //   average_err += fabs(mm_res[i] / 2.0 - matrix_4096x64_T[0][i]);
  //   // printf("%+.10lf %+.10lf\n", mm_res[i] / 2.0, matrix_4096x64_T[0][i]);
  // }
  // std::cout << "average_err: " << average_err / 4096.0 << std::endl;

  vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};
  vector<double> output;
  vector<double> temp;

  PhantomPlaintext input_plain;
  PhantomCiphertext input_cipher;
  PhantomPlaintext temp_plain;
  PhantomCiphertext temp_cipher;
  PhantomCiphertext output_cipher;
  PhantomPlaintext output_plain;

  ckks_evaluator.encoder.encode(input, SCALE, input_plain);
  ckks_evaluator.encryptor.encrypt(input_plain, input_cipher);

  mme.multiply_power_of_x(input_cipher, output_cipher, 1);

  ckks_evaluator.decryptor.decrypt(output_cipher, output_plain);
  ckks_evaluator.encoder.decode(output_plain, output);
  
  for (auto i = 0; i < 7; i++) {
    cout << fixed << setprecision(5) << output[i] << ", ";
  }
  cout << endl;
}

int main() {
  /*
    MatMul
  */
  if (TEST_TARGET == "MatMul") {
    MM_test();
    return 0;
  }

  EncryptionParameters params(scheme_type::ckks);

  params.set_poly_modulus_degree(N);
  params.set_coeff_modulus(CoeffModulus::Create(N, COEFF_MODULI));

  PhantomContext context(params);

  PhantomSecretKey secret_key(context);
  PhantomPublicKey public_key = secret_key.gen_publickey(context);
  PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
  PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

  PhantomCKKSEncoder encoder(context);

  CKKSEvaluator ckks_evaluator(context, public_key, secret_key, encoder, relin_keys, galois_keys, SCALE);

  vector<double> input;
  PhantomPlaintext plain_input;
  PhantomCiphertext cipher_input;
  PhantomCiphertext cipher_output;
  PhantomPlaintext plain_output;
  vector<double> output;

  /*
    GELU
  */
  if (TEST_TARGET == "GELU") {
    GELUEvaluator gelu_evaluator(ckks_evaluator);

    double num;
    vector<double> gelu_calibration;
    ifstream input_file("../../data/input/gelu_input_32768.txt");
    while (input_file >> num) {
      input.push_back(num);
    }
    input_file.close();

    ifstream calibration_file("../../data/calibration/gelu_calibration_32768.txt");
    while (calibration_file >> num) {
      gelu_calibration.push_back(num);
    }
    calibration_file.close();

    ckks_evaluator.encoder.encode(input, SCALE, plain_input);
    ckks_evaluator.encryptor.encrypt(plain_input, cipher_input);

    auto timer = Timer();
    gelu_evaluator.gelu(cipher_input, cipher_output);
    timer.stop();

    cout << "[GELU] 32768 takes: " << timer.duration() << " milliseconds" << endl;
    cout << "Mean Absolute Error: " << ckks_evaluator.calculate_MAE(gelu_calibration, cipher_output, N / 2) << endl;
  }

  /*
    LayerNorm
  */
  if (TEST_TARGET == "LayerNorm") {
    LNEvaluator ln_evaluator(ckks_evaluator);

    double num;
    vector<double> input, layernorm_calibration;
    ifstream input_file("../../data/input/layernorm_input_16_768.txt");
    while (input_file >> num) {
      input.push_back(num);
    }
    input_file.close();

    ifstream calibration_file("../../data/calibration/layernorm_calibration_16_768.txt");
    while (calibration_file >> num) {
      layernorm_calibration.push_back(num);
    }
    calibration_file.close();

    ckks_evaluator.encoder.encode(input, SCALE, plain_input);
    ckks_evaluator.encryptor.encrypt(plain_input, cipher_input);

    auto timer = Timer();
    ln_evaluator.layer_norm(cipher_input, cipher_output, 1024);
    timer.stop();

    cout << "[LayerNorm] 16 x 768 takes: " << timer.duration() << " milliseconds" << endl;
    cout << "Mean Absolute Error: " << ckks_evaluator.calculate_MAE(layernorm_calibration, cipher_output, 768) << endl;
  }

  /*
    Softmax
  */
  if (TEST_TARGET == "SoftMax") {
    SoftmaxEvaluator softmax_evaluator(ckks_evaluator);

    double num;
    vector<double> input, softmax_calibration;
    ifstream input_file("../../data/input/softmax_input_128_128.txt");
    while (input_file >> num) {
      input.push_back(num);
    }
    input_file.close();

    ifstream calibration_file("../../data/calibration/softmax_calibration_128_128.txt");
    while (calibration_file >> num) {
      softmax_calibration.push_back(num);
    }
    calibration_file.close();

    ckks_evaluator.encoder.encode(input, SCALE, plain_input);
    ckks_evaluator.encryptor.encrypt(plain_input, cipher_input);

    auto timer = Timer();
    softmax_evaluator.softmax(cipher_input, cipher_output, 128);
    timer.stop();

    cout << "[Softmax] 128 x 128 takes: " << timer.duration() << " milliseconds" << endl;
    cout << "Mean Absolute Error: " << ckks_evaluator.calculate_MAE(softmax_calibration, cipher_output, 128) << endl;
  }
}
