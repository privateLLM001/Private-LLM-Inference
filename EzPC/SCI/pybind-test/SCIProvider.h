#pragma once
#include <string>
#include <vector>
#include <library_fixed.h>

#if not defined(PARTY_ALICE) and not defined(PARTY_BOB)
    #define PARTY_ALICE
#endif

#if not defined(SCI_HE) and not defined(SCI_OT)
    #define SCI_HE
#endif

#if not defined(BIT_LENGTH)
    #define BIT_LENGTH 59
#endif

#ifdef PARTY_BOB
int party = 2;
#else
int party = 1;
#endif

int port = 32001;
std::string address = "127.0.0.1";
int num_threads = 4;
int32_t bitlength = BIT_LENGTH;

class SCIProvider {

    int sf;

public:

    SCIProvider(int sf): sf(sf) {}

    void startComputation() {
        StartComputation();
    }

    void endComputation() {
        EndComputation(false);
    }

    int dbits() {
        return bitlength - sf;
    }

    std::vector<uint64_t> sqrt(std::vector<uint64_t> share, int64_t scale_in, int64_t scale_out, bool inverse) {
        long n = share.size();
        std::vector<uint64_t> ret(n);
        Sqrt(1, n, scale_in, scale_out, bitlength, bitlength, inverse, share.data(), ret.data());
        return ret;
    }

    std::vector<uint64_t> elementwise_multiply(std::vector<uint64_t> share1, std::vector<uint64_t> share2) {
        long n = share1.size();
        std::vector<uint64_t> ret(n);
        ElemWiseSecretSharedVectorMult(n, share1.data(), share2.data(), ret.data());
        return ret;
    }

    // Exp only accepts negative inputs
    std::vector<uint64_t> exp(std::vector<uint64_t> base) {
        long n = base.size();
        std::vector<uint64_t> ret(n);
        Exp(base.data(), ret.data(), 1, n, bitlength, bitlength, sf, sf);
        return ret;
    }

    std::vector<uint64_t> exp_reduce(std::vector<uint64_t> base) {
        long n = base.size();
        std::vector<uint64_t> ret(n);
        Exp(base.data(), ret.data(), 1, n, bitlength, bitlength - sf, sf, sf);
        return ret;
    }

    std::vector<uint64_t> tanh(std::vector<uint64_t> share) {
        long n = share.size();
        std::vector<uint64_t> ret(n);
        TanH(1, n, 1<<sf, 1<<sf, bitlength, bitlength, share.data(), ret.data());
        return ret;
    }

    std::vector<uint64_t> div(std::vector<uint64_t> nom, std::vector<uint64_t> den) {
        long n = nom.size();
        long mask = (1ull << (bitlength - sf)) - 1;
        for (size_t i = 0; i < n; i++) {
            nom[i] = nom[i] & mask;
            den[i] = den[i] & mask;
        }
        std::vector<uint64_t> ret(n);
        // Div(1, n, sf, sf, sf, bitlength - sf, (int64_t*)exped.data(), (int64_t*)summed.data(), (int64_t*)ret.data());
        Div(nom.data(), den.data(), ret.data(), n, 1, bitlength - sf, bitlength - sf, bitlength - sf, sf, sf, sf);
        for (size_t i = 0; i < n; i++) {
            ret[i] = ret[i] << sf;
        }
        return ret;
    }

    std::vector<uint64_t> softmax(std::vector<uint64_t> share, size_t dims) {
        long n = share.size();
        uint64_t mod = (1ull << bitlength);
        // for (size_t i = 0; i < n; i++) {
        //     share[i] = (share[i] + mod - (2ull << sf)) % mod;
        // }
        std::vector<uint64_t> exped(n);
        Exp(share.data(), exped.data(), 1, n, bitlength, bitlength - sf, sf, sf);
        mod = 1ull << (bitlength - sf);
        for (size_t i = 0; i < n; i++) {
            // exped[i] = (exped[i] * 4ull) % mod;
        }
        std::vector<uint64_t> summed(n);
        long m = n / dims;
        for (size_t i = 0; i < m; i++) {
            summed[i * dims] = 0;
            for (size_t j = 0; j < dims; j++) {
                summed[i * dims] = (summed[i * dims] + exped[i * dims + j]) % mod;
            }
            for (size_t j = 1; j < dims; j++) {
                summed[i * dims + j] = summed[i * dims];
            }
        }
        std::vector<uint64_t> ret(n);

        // Div(1, n, sf, sf, sf, bitlength - sf, (int64_t*)exped.data(), (int64_t*)summed.data(), (int64_t*)ret.data());
        Div(exped.data(), summed.data(), ret.data(), n, 1, bitlength - sf, bitlength - sf, bitlength - sf, sf, sf, sf);
        for (size_t i = 0; i < n; i++) {
            ret[i] = ret[i] << sf;
        }
        return ret;
    }

};