#include "matrix_mul.h"
#include <cmath>
#include <seal/ciphertext.h>
#include <seal/plaintext.h>
#include <vector>

using namespace std;
using namespace seal;
using namespace std::chrono;
using namespace seal::util;

vector<Ciphertext> MMEvaluator::expand_ciphertext(
    const Ciphertext &encrypted, uint32_t m, GaloisKeys &galkey, vector<uint32_t> &galois_elts)
{
    uint32_t logm = ceil(log2(m));
    Plaintext two("2");
    auto n = ckks->N;
    vector<Ciphertext> temp;
    temp.push_back(encrypted);
    Ciphertext tempctxt;
    Ciphertext tempctxt_rotated;
    Ciphertext tempctxt_shifted;
    Ciphertext tempctxt_rotatedshifted;

    for (uint32_t i = 0; i < logm; i++) {
        vector<Ciphertext> newtemp(temp.size() << 1);
        int index_raw = (n << 1) - (1 << i);
        int index = (index_raw * galois_elts[i]) % (n << 1);
        for (uint32_t a = 0; a < temp.size(); a++) {
            ckks->evaluator->apply_galois(temp[a], ckks->rots[i], *(ckks->galois_keys), tempctxt_rotated); // sub
            ckks->evaluator->add(temp[a], tempctxt_rotated, newtemp[a]);
            multiply_power_of_X(temp[a], tempctxt_shifted, index_raw); // x**-1
            multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index);
            ckks->evaluator->add(tempctxt_shifted, tempctxt_rotatedshifted, newtemp[a + temp.size()]);
        }
        temp = newtemp;
    }
    return temp;
}

void MMEvaluator::multiply_power_of_X(Ciphertext &encrypted, Ciphertext &destination, int index)
{
    string s = "";
    destination = encrypted;
    while (index >= ckks->N - 1) {
        s = "1x^" + to_string(ckks->N - 1);
        Plaintext p(s);
        ckks->evaluator->multiply_plain(destination, p, destination);
        index -= ckks->N - 1;
    }

    s = "1x^" + to_string(index);

    Plaintext p(s);
    ckks->evaluator->multiply_plain(destination, p, destination);
}

void MMEvaluator::matrix_mul(vector<vector<double>> &x, vector<vector<double>> &y, vector<Ciphertext> &res)
{
    chrono::high_resolution_clock::time_point time_start, time_end;

    vector<Plaintext> a_pts;
    a_pts.reserve(768);
    for (int i = 0; i < 768; i++) {
        Plaintext pt;
        ckks->encoder->encode(x[i], ckks->scale, pt);
        a_pts.emplace_back(pt);
    }

    vector<Ciphertext> b_compressed_cts;
    for (int i = 0; i < 768 * 768 / ckks->N; i++) {
        Plaintext pt;
        Ciphertext ct;
        expandEncode(y[i], ct);
        b_compressed_cts.push_back(ct);
    }

    time_start = high_resolution_clock::now();
    vector<Ciphertext> b_expanded_cts;
    for (Ciphertext &ct : b_compressed_cts) {
        vector<Ciphertext> temp_cts = expand_ciphertext(ct, ckks->N, *ckks->galois_keys, ckks->rots);
        cout << "expanding..." << endl;
        b_expanded_cts.insert(
            b_expanded_cts.end(), make_move_iterator(temp_cts.begin()), make_move_iterator(temp_cts.end()));
    }
    time_end = high_resolution_clock::now();
    cout << "expanding time: " << duration_cast<std::chrono::seconds>(time_end - time_start).count() << "seconds"
         << endl;

    vector<Ciphertext> res_cts;
    Plaintext pt(ckks->N, 0);
    time_start = high_resolution_clock::now();
    for (int i = 0; i < 768; i++) {
        Ciphertext res_col_ct;
        ckks->encryptor->encrypt(pt, res_col_ct);
        for (int j = 0; j < 768; j++) {
            Ciphertext temp;
            ckks->evaluator->multiply_plain(b_expanded_cts[i * 768 + j], a_pts[j], temp);
            ckks->evaluator->add(res_col_ct, temp, res_col_ct);
        }
        res_cts.push_back(res_col_ct);
    }
    time_end = high_resolution_clock::now();
    cout << "calculating res time: " << duration_cast<seconds>(time_end - time_start).count() << "seconds" << endl;
}

void MMEvaluator::expandEncode(vector<double> &val, Ciphertext &ct)
{
    Plaintext zero_pt;
    ckks->encoder->encode(std::vector<double>(ckks->N / 2, 0), ckks->scale, zero_pt);
    Ciphertext zero;
    ckks->encryptor->encrypt(zero_pt, zero);

    auto context = *ckks->context;
    auto context_data = context.get_context_data(context.first_parms_id());
    auto param = context_data->parms();
    auto ntt_tables = context_data->small_ntt_tables();

    auto poly_modulus_degree = ckks->N;

    Plaintext p(poly_modulus_degree * 2);

    for (auto i = 0; i < poly_modulus_degree; i++) {
        val[i] = 10.0 * 2.0 * (1.0 * rand() / RAND_MAX - 0.5);
    }
    for (auto i = 0; i < poly_modulus_degree; i++) {
        auto coeffd = std::round(val[i] * 10000000000);
        bool is_negative = std::signbit(coeffd);
        auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));
        if (is_negative) {
            for (std::size_t j = 0; j < 2; j++) {
                p[i + (j * poly_modulus_degree)] = util::negate_uint_mod(
                    util::barrett_reduce_64(coeffu, param.coeff_modulus()[j]), param.coeff_modulus()[j]);
            }
        } else {
            for (std::size_t j = 0; j < 2; j++) {
                p[i + (j * poly_modulus_degree)] = util::barrett_reduce_64(coeffu, param.coeff_modulus()[j]);
            }
        }
    }
    for (std::size_t i = 0; i < 2; i++) {
        util::ntt_negacyclic_harvey(p.data(i * poly_modulus_degree), ntt_tables[i]);
    }
    p.parms_id() = context.first_parms_id();
    p.scale() = 10000000000;

    zero.scale() = p.scale();

    ckks->evaluator->add_plain(zero, p, ct);
}
