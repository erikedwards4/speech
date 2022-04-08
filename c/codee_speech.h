#pragma once

#include "codee_math.h"
#include "codee_dsp.h"
#include "codee_aud.h"


#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int kaldi_fbank_s (float *Y, float *X, const size_t N, const float sr, const float frame_length, const float frame_shift, const int snip_edges, const float dither, const int dc0, const int raw_energy, const float preemph, const char win_type[], const int amp, const float lof, const float hif, const size_t B, const int lg, const int use_energy, const int mn0);
int kaldi_fbank_d (double *Y, double *X, const size_t N, const double sr, const double frame_length, const double frame_shift, const int snip_edges, const double dither, const int dc0, const int raw_energy, const double preemph, const char win_type[], const int amp, const double lof, const double hif, const size_t B, const int lg, const int use_energy, const int mn0);
int kaldi_fbank_default_s (float *Y, float *X, const size_t N, const float frame_shift, const int snip_edges, const float dither, const int dc0, const int raw_energy, const float preemph, const char win_type[], const int amp, const float lof, const float hif, const int lg, const int use_energy, const int mn0);
int kaldi_fbank_default_d (double *Y, double *X, const size_t N, const double frame_shift, const int snip_edges, const double dither, const int dc0, const int raw_energy, const double preemph, const char win_type[], const int amp, const double lof, const double hif, const int lg, const int use_energy, const int mn0);

int kaldi_mfcc_s (float *Y, float *X, const size_t N, const float sr, const float frame_length, const float frame_shift, const int snip_edges, const float dither, const int dc0, const int raw_energy, const float preemph, const char win_type[], const float lof, const float hif, const size_t B, const size_t C, const float Q, const int use_energy, const int mn0);
int kaldi_mfcc_d (double *Y, double *X, const size_t N, const double sr, const double frame_length, const double frame_shift, const int snip_edges, const double dither, const int dc0, const int raw_energy, const double preemph, const char win_type[], const double lof, const double hif, const size_t B, const size_t C, const double Q, const int use_energy, const int mn0);

int kaldi_plp_s (float *Y, float *X, const size_t N, const float sr, const float frame_length, const float frame_shift, const int snip_edges, const float dither, const int dc0, const int raw_energy, const float preemph, const char win_type[], const float lof, const float hif, const size_t B, const float compress, const size_t C, const float Q, const float cep_scale, const int use_energy, const int mn0);
int kaldi_plp_d (double *Y, double *X, const size_t N, const double sr, const double frame_length, const double frame_shift, const int snip_edges, const double dither, const int dc0, const int raw_energy, const double preemph, const char win_type[], const double lof, const double hif, const size_t B, const double compress, const size_t C, const double Q, const double cep_scale, const int use_energy, const int mn0);

int kaldi_spectrogram_s (float *Y, float *X, const size_t N, const float sr, const float frame_length, const float frame_shift, const int snip_edges, const float dither, const int dc0, const int raw_energy, const float preemph, const char win_type[], const int mn0);
int kaldi_spectrogram_d (double *Y, double *X, const size_t N, const double sr, const double frame_length, const double frame_shift, const int snip_edges, const double dither, const int dc0, const int raw_energy, const double preemph, const char win_type[], const int mn0);

#ifdef __cplusplus
}
}
#endif
