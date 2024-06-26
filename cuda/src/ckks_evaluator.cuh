#pragma once

#include "phantom.h"

using namespace std;
using namespace phantom;

namespace nexus {
class Encoder {
 private:
  PhantomContext *context;
  PhantomCKKSEncoder *encoder;

 public:
  Encoder() = default;

  Encoder(PhantomContext &context, PhantomCKKSEncoder &encoder) {
    this->context = &context;
    this->encoder = &encoder;
  }

  // Vector inputs
  inline void encode(vector<double> values, size_t chain_index, double scale, PhantomPlaintext &plain) {
    encoder->encode(*context, values, scale, plain, chain_index);
  }

  inline void encode(vector<double> values, double scale, PhantomPlaintext &plain) {
    encoder->encode(*context, values, scale, plain);
  }

  // Value inputs (fill all slots with that value)
  inline void encode(double value, size_t chain_index, double scale, PhantomPlaintext &plain) {
    vector<double> values(encoder->slot_count(), value);
    encoder->encode(*context, values, scale, plain, chain_index);
  }

  inline void encode(double value, double scale, PhantomPlaintext &plain) {
    vector<double> values(encoder->slot_count(), value);
    encoder->encode(*context, values, scale, plain);
  }

  inline void decode(PhantomPlaintext &plain, vector<double> &values) {
    encoder->decode(*context, plain, values);
  }
};

class Encryptor {
 private:
  PhantomContext *context;
  PhantomPublicKey *encryptor;

 public:
  Encryptor() = default;

  Encryptor(PhantomContext &context, PhantomPublicKey &encryptor) {
    this->context = &context;
    this->encryptor = &encryptor;
  }

  inline void encrypt(PhantomPlaintext &plain, PhantomCiphertext &ct) {
    encryptor->encrypt_asymmetric(*context, plain, ct);
  }
};

class Evaluator {
 private:
  PhantomContext *context;

 public:
  Evaluator() = default;
  Evaluator(PhantomContext &context) : context(&context) {}

  // Mod switch
  void mod_switch_to_next(const PhantomCiphertext &encrypted, PhantomCiphertext &destination);

  inline void mod_switch_to_next_inplace(PhantomCiphertext &ct) {
    // PhantomCiphertext dest = ct;
    // mod_switch_to_next(ct, dest);
    // ct = dest;
    ::mod_switch_to_next_inplace(*context, ct);
  }

  inline void mod_switch_to_inplace(PhantomCiphertext &ct, size_t chain_index) {
    // if (ct.chain_index() > chain_index) {
    //   throw std::invalid_argument("cannot switch to higher level modulus");
    // }

    // PhantomCiphertext destination = ct;

    // while (destination.chain_index() != chain_index) {
    //   mod_switch_to_next_inplace(destination);
    // }

    // ct = destination;
    ::mod_switch_to_inplace(*context, ct, chain_index);
  }

  inline void mod_switch_to_inplace(PhantomPlaintext &pt, size_t chain_index) {
    ::mod_switch_to_inplace(*context, pt, chain_index);
  }

  // Rescale
  inline void rescale_to_next_inplace(PhantomCiphertext &ct) {
    ::rescale_to_next_inplace(*context, ct);
  }

  // Relinearization
  inline void relinearize_inplace(PhantomCiphertext &ct, PhantomRelinKey &relin_keys) {
    ::relinearize_inplace(*context, ct, relin_keys);
  }

  // Multiplication
  inline void square(PhantomCiphertext &ct, PhantomCiphertext &dest) {
    dest = ::multiply(*context, ct, ct);
  }

  inline void multiply(PhantomCiphertext &ct1, PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    // PhantomCiphertext tmp(ct1);
    // ::multiply_inplace(*context, tmp, ct2);
    // dest = tmp;
    dest = ::multiply(*context, ct1, ct2);
  }

  inline void multiply_inplace(PhantomCiphertext &ct1, PhantomCiphertext &ct2) {
    ::multiply_inplace(*context, ct1, ct2);
  }

  inline void multiply_plain(PhantomCiphertext &ct, PhantomPlaintext &plain, PhantomCiphertext &dest) {
    dest = ::multiply_plain(*context, ct, plain);
  }

  inline void multiply_plain_inplace(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    ::multiply_plain_inplace(*context, ct, plain);
  }

  // Addition
  inline void add_plain(PhantomCiphertext &ct, PhantomPlaintext &plain, PhantomCiphertext &dest) {
    dest = ::add_plain(*context, ct, plain);
  }

  inline void add_plain_inplace(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    ::add_plain_inplace(*context, ct, plain);
  }

  inline void add(PhantomCiphertext &ct1, PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    dest = ::add(*context, ct1, ct2);
  }

  inline void add_inplace(PhantomCiphertext &ct1, PhantomCiphertext &ct2) {
    ::add_inplace(*context, ct1, ct2);
  }

  // Subtraction
  inline void sub_plain(PhantomCiphertext &ct, PhantomPlaintext &plain, PhantomCiphertext &dest) {
    dest = ::sub_plain(*context, ct, plain);
  }

  inline void sub(PhantomCiphertext &ct1, PhantomCiphertext &ct2, PhantomCiphertext &dest) {
    dest = ::sub(*context, ct1, ct2);
  }

  // Rotation
  inline void rotate_vector(PhantomCiphertext &ct, int steps, PhantomGaloisKey &galois_keys, PhantomCiphertext &dest) {
    dest = ::rotate_vector(*context, ct, steps, galois_keys);
  }

  // Negation
  inline void negate(PhantomCiphertext &ct, PhantomCiphertext &dest) {
    dest = ::negate(*context, ct);
  }
};

class Decryptor {
 private:
  PhantomContext *context;
  PhantomSecretKey *decryptor;

 public:
  Decryptor() = default;
  Decryptor(PhantomContext &context, PhantomSecretKey &decryptor) {
    this->context = &context;
    this->decryptor = &decryptor;
  }

  inline void decrypt(PhantomCiphertext &ct, PhantomPlaintext &plain) {
    decryptor->decrypt(*context, ct, plain);
  }
};

class CKKSEvaluator {
 private:
  // Sign function g,f coefficients
  vector<double> g4_coeffs = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
  vector<double> g4_coeffs_last;
  vector<double> f4_coeffs = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
  vector<double> f4_coeffs_last;

  // Helper functions
  uint64_t get_modulus(PhantomCiphertext &x, int k);
  void re_encrypt(PhantomCiphertext &ct);

  PhantomCiphertext init_guess(PhantomCiphertext x);
  PhantomCiphertext eval_line(PhantomCiphertext x, PhantomPlaintext m, PhantomPlaintext c);

  // Evaluation functions
  PhantomCiphertext newton_iter(PhantomCiphertext x, PhantomCiphertext res, int iter);
  pair<PhantomCiphertext, PhantomCiphertext> goldschmidt_iter(PhantomCiphertext v, PhantomCiphertext y, int d = 1);
  void eval_odd_deg9_poly(vector<double> &a, PhantomCiphertext &x, PhantomCiphertext &dest);

 public:
  PhantomContext *context;
  Encryptor encryptor;
  Decryptor decryptor;
  Encoder encoder;
  Evaluator evaluator;
  PhantomRelinKey *relin_keys;
  PhantomGaloisKey *galois_keys;

  double scale;
  size_t slot_count;

  CKKSEvaluator(PhantomContext &context, PhantomPublicKey &encryptor, PhantomSecretKey &decryptor,
                PhantomCKKSEncoder &encoder, PhantomRelinKey &relin_keys, PhantomGaloisKey &galois_keys,
                double scale) {
    this->context = &context;
    this->relin_keys = &relin_keys;
    this->galois_keys = &galois_keys;

    this->scale = scale;
    this->slot_count = encoder.slot_count();

    Encoder ckks_encoder(context, encoder);
    this->encoder = ckks_encoder;

    Encryptor ckks_encryptor(context, encryptor);
    this->encryptor = ckks_encryptor;

    Evaluator ckks_evaluator(context);
    this->evaluator = ckks_evaluator;

    Decryptor ckks_decryptor(context, decryptor);
    this->decryptor = ckks_decryptor;
  }

  // Helper functions
  vector<double> init_vec_with_value(int N, double init_value);

  // Evaluation functions
  PhantomCiphertext sgn_eval2(PhantomCiphertext x, int d_g, int d_f);
  PhantomCiphertext invert_sqrt(PhantomCiphertext x, int d_newt = 20, int d_gold = 1);

  // Metrics calcuation functions
  double calculate_MAE(vector<double> &y_true, PhantomCiphertext &ct);
};
}  // namespace nexus
