//Kaldi-format MFCC features
//STFT (short-term Fourier transform) of univariate time series X,
//obtains power on a mel frequency scale in Y, and then does
//DCT and lifter to obtain mel-frequency cepstral coefficients (MFCCs)

//This takes a univariate time series (vector) X of length N,
//and outputs C MFCCs at each of W windows.
//If Y is row-major, then it has size W x C.
//If Y is col-major, then it has size C x W.
//where C is the number of cepstral coefficients.

//This follows Kaldi and torchaudio conventions for compatibility.

//The following params and boolean options are included:
//N:            size_t  length of input signal X in samps
//sr:           float   sample rate in Hz
//frame_length: float   length of each frame (window) in msec (often 25 ms)
//frame_shift:  float   step size between each frame center in msec (often 10 ms)
//snip_edges:   bool    controls style of framing w.r.t. edges of X
//dither:       float   dithering weight (set to 0.0 for no dither)
//dc0:          bool    to subtract mean from each frame after dither (usually recommended)
//raw_energy:   bool    compute raw energy of each frame before preemph and window (instead of after)
//preemph:      float   preemph coeff (set to 0.0 for no preemph)
//win_type:     string  window type in {rectangular,blackman,hamming,hann,povey} 
//lof:          float   low freq cutoff for mel bins (typical default is 20 Hz)
//hif:          float   high freq cutoff for mel bins (typical default is Nyquist)
//B:            size_t  num mel bins (typical default is 23)
//C:            size_t  num cepstral coeffs (typical default is 13)
//Q:            float   lifter coefficient Q (typical default is 22.0)
//use_energy:   bool    use raw_energy for first feature (feat dim remains C)
//mn0:          bool    to subtract means of C feats in Y before output (not usually recommended)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <fftw3.h>
#include "codee_speech.h"

#ifndef M_PI
   #define M_PI 3.141592653589793238462643383279502884
#endif

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int kaldi_mfcc_s (float *Y, float *X, const size_t N, const float sr, const float frame_length, const float frame_shift, const int snip_edges, const float dither, const int dc0, const int raw_energy, const float preemph, const char win_type[], const float lof, const float hif, const size_t B, const size_t C, const float Q, const int use_energy, const int mn0)
{
    if (N<1u) { fprintf(stderr,"error in kaldi_mfcc_s: N (nsamps in signal) must be positive\n"); return 1; }
    if (sr<FLT_EPSILON) { fprintf(stderr,"error in kaldi_mfcc_s: sr must be positive\n"); return 1; }
    if (frame_length<FLT_EPSILON) { fprintf(stderr,"error in kaldi_mfcc_s: frame_length must be positive\n"); return 1; }
    if (frame_shift<FLT_EPSILON) { fprintf(stderr,"error in kaldi_mfcc_s: frame_shift must be positive\n"); return 1; }
    if (dither<0.0f) { fprintf(stderr,"error in kaldi_mfcc_s: dither must be nonnegative\n"); return 1; }
    if (preemph<0.0f) { fprintf(stderr,"error in kaldi_mfcc_s: preemph coeff must be nonnegative\n"); return 1; }
    if (lof<0.0f) { fprintf(stderr,"error in kaldi_mfcc_s: lof (low-freq cutoff) must be nonnegative\n"); return 1; }
    if (hif<=lof) { fprintf(stderr,"error in kaldi_mfcc_s: hif (high-freq cutoff) must be > lof\n"); return 1; }
    if (B<1u) { fprintf(stderr,"error in kaldi_mfcc_s: B (num mel bins) must be positive\n"); return 1; }
    if (C<1u) { fprintf(stderr,"error in kaldi_mfcc_s: C (num cepstral coeffs) must be positive\n"); return 1; }
    if (C>B) { fprintf(stderr,"error in kaldi_mfcc_s: C (num cepstral coeffs) must be <= B (num mel bins)\n"); return 1; }
    if (Q<0.0f) { fprintf(stderr,"error in kaldi_mfcc_s: Q (lifter coeff) must be nonnegative\n"); return 1; }

    //Set L (frame_length in samps)
    const size_t L = (size_t)(sr*frame_length/1000.0f);
    if (L<1u) { fprintf(stderr,"error in kaldi_mfcc_s: frame_length must be greater than 1 sample\n"); return 1; }
    if (snip_edges && L>N) { fprintf(stderr,"error in kaldi_mfcc_s: frame length must be < signal length if snip_edges\n"); return 1; }

    //Set stp (frame_shift in samps)
    const size_t stp = (size_t)(sr*frame_shift/1000.0f);
    if (stp<1u) { fprintf(stderr,"error in kaldi_mfcc_s: frame_shift must be greater than 1 sample\n"); return 1; }

    //Set W (number of frames or windows)
    const size_t W = (snip_edges) ? 1u+(N-L)/stp : (N+stp/2u)/stp;
    if (W==0u) { return 0; }

    //Initialize framing
    const size_t Lpre = L/2u;                   //nsamps before center samp
    int ss = (int)(stp/2u) - (int)Lpre;         //start-samp of current frame
    int n, prev_n = 0;                          //current/prev samps in X
    const int xd = (int)L - (int)stp;           //a fixed increment after each frame for speed below

    //Initialize dither (this is a direct randn generator using method of PCG library)
    const float M_2PI = (float)(2.0*M_PI);
    float u1, u2, R;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;
    if (dither>FLT_EPSILON)
    {
        //Init random num generator
        if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in kaldi_mfcc_s: problem with timespec_get.\n"); perror("timespec_get"); return 1; }
        state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;
    }

    //Initialize raw_energy
    const float rawe_floor = FLT_EPSILON;
    float rawe = 0.0f;

    //Get win (window vec of length L)
    float *win;
    if (!(win=(float *)malloc(L*sizeof(float)))) { fprintf(stderr,"error in kaldi_mfcc_s: problem with malloc. "); perror("malloc"); return 1; }
    if (strncmp(win_type,"rectangular",11u)==0)
    {
        for (size_t l=0u; l<L; ++l) { win[l] = 1.0f; }
    }
    else
    {
        const float a = (float)(2.0*M_PI/(double)(L-1u));
        if (strncmp(win_type,"blackman",8u)==0)
        {
            const float coeff = 0.42f, coeff1 = 1.0f - coeff; //blackman_coeff
            for (size_t l=0u; l<L; ++l) { win[l] = coeff - 0.5f*cosf((float)l*a) + coeff1*cosf((float)(2u*l)*a); }
        }
        else if (strncmp(win_type,"hamming",7u)==0)
        {
            for (size_t l=0u; l<L; ++l) { win[l] = 0.54f - 0.46f*cosf((float)l*a); }
        }
        else if (strncmp(win_type,"hann",4u)==0)
        {
            for (size_t l=0u; l<L; ++l) { win[l] = 0.5f - 0.5f*cosf((float)l*a); }
        }
        else if (strncmp(win_type,"povey",5u)==0)
        {
            for (size_t l=0u; l<L; ++l) { win[l] = powf(0.5f-0.5f*cosf((float)l*a),0.85f); }
        }
        else
        {
            fprintf(stderr,"error in kaldi_mfcc_s: window type must be rectangular, blackman, hamming, hann or povey (lower-cased)\n"); return 1;
        }
    }

    //Set nfft (FFT transform length), which is next pow2 of L
    size_t nfft = 1u;
    while (nfft<L) { nfft *= 2u; }

    //Set F (num non-negative FFT freqs)
    const size_t F = nfft/2u + 1u;
    
    //Initialize FFT (using lib fftw3)
    float *Xw, *Yw;
    Xw = (float *)fftwf_malloc(nfft*sizeof(float));
    Yw = (float *)fftwf_malloc(nfft*sizeof(float));
    fftwf_plan fft_plan = fftwf_plan_r2r_1d((int)nfft,Xw,Yw,FFTW_R2HC,FFTW_ESTIMATE);
    if (!fft_plan) { fprintf(stderr,"error in kaldi_mfcc_s: problem creating fftw plan"); return 1; }
    for (size_t nf=nfft; nf>0u; --nf, ++Xw) { *Xw = 0.0f; }
    Xw -= nfft;

    //Initialize Yf (initial output with power at F STFT freqs)
    float *Yf;
    if (!(Yf=(float *)malloc(F*sizeof(float)))) { fprintf(stderr,"error in kaldi_mfcc_s: problem with malloc. "); perror("malloc"); return 1; }

    //Initialize Hz-to-mel transfrom matrix (F2B)
    const float finc = sr/(float)nfft;                          //freq increment in Hz for FFT freqs
    const float lomel = 1127.0f * logf(1.0f+lof/700.0f);        //low-mel cutoff
    const float himel = 1127.0f * logf(1.0f+hif/700.0f);        //high-mel cutoff
    const float dmel = (himel-lomel) / (float)(B+1u);           //controls spacing on mel scale
    float lmel, cmel, rmel;                                     //current left, center, right mels
    float *mels;                                                //maps STFT freqs to mels
    float *F2B;                                                 //transform matrix for STFT power in F freqs to B mel bins
    size_t *beg_f, *num_f, nf2b = 0u;                           //indices and size of F2B as sparse matrix
    mels = (float *)malloc(F*sizeof(float));
    F2B = (float *)malloc(B*(F/2u)*sizeof(float));
    beg_f = (size_t *)malloc(B*sizeof(size_t));
    num_f = (size_t *)malloc(B*sizeof(size_t));
    if (!mels) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!F2B) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!beg_f) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!num_f) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    for (size_t f=0; f<F; ++f) { mels[f] = 1127.0f * logf(1.0f+(float)f*finc/700.0f); }
    lmel = lomel; cmel = lmel + dmel; rmel = cmel + dmel;
    for (size_t b=B; b>0u; --b)
    {
        size_t f = 0u;
        while (*mels<lmel && f<F) { ++mels; ++f; }
        *beg_f = f;
        while (*mels<cmel && f<F) { *F2B++ = (*mels-lmel)/dmel; ++mels; ++f; }
        while (*mels<rmel && f<F) { *F2B++ = (rmel-*mels)/dmel; ++mels; ++f; }
        *num_f = f - *beg_f++; nf2b += *num_f++;
        mels -= f;
        lmel = cmel; cmel = rmel;
        rmel = (b==2u) ? himel : rmel + dmel;
    }
	F2B -= nf2b; beg_f -= B; num_f -= B;

    //Initialize DCT
    //Kaldi uses a dct_matrix, that is B x B, and then resized to nceps x B.
    float *Xd, *Yd;
    Xd = (float *)fftwf_malloc(B*sizeof(float));
    Yd = (float *)fftwf_malloc(B*sizeof(float));
    fftwf_plan dct_plan = fftwf_plan_r2r_1d((int)B,Xd,Yd,FFTW_REDFT10,FFTW_ESTIMATE);
    if (!dct_plan) { fprintf(stderr,"error in kaldi_mfcc_s: problem creating fftw plan"); return 1; }
    for (size_t b=B; b>0u; --b, ++Xd) { *Xd = 0.0f; }
    Xd -= B;

    //Initialize lifter (include DCT scaling)
    const float sc = 1.0f/sqrtf((float)(2u*B));
    float *lift;
    if (!(lift=(float *)malloc(C*sizeof(float)))) { fprintf(stderr,"error in kaldi_mfcc_s: problem with malloc. "); perror("malloc"); return 1; }
    if (Q>FLT_EPSILON)
    {
        for (size_t c=0u; c<C; ++c, ++lift) { *lift = 1.0f + 0.5f*Q*sinf((float)M_PI*(float)c/Q); }
    }
    else { for (size_t c=0u; c<C; ++c, ++lift) { *lift = 1.0f; } }
    for (size_t c=1u; c<C; ++c) { *--lift *= sc; }
    *--lift *= 0.5f/sqrtf((float)B);      //proper DC scaling

    //Process each of W frames
    for (size_t w=W; w>0u; --w)
    {
        //Copy one frame of X into Xw
        if (snip_edges)
        {
            for (size_t l=L; l>0u; --l, ++X, ++Xw) { *Xw = *X; }
            X -= xd;
        }
        else
        {
            if (ss<0 || ss>(int)N-(int)L)
            {
                for (int s=ss; s<ss+(int)L; ++s, ++Xw)
                {
                    n = s; //This ensures extrapolation by signal reversal to any length
                    while (n<0 || n>=(int)N) { n = (n<0) ? -n-1 : (n<(int)N) ? n : 2*(int)N-1-n; }
                    X += n - prev_n; prev_n = n;
                    *Xw = *X;
                }
            }
            else
            {
                X += ss - prev_n;
                for (size_t l=L; l>0u; --l, ++X, ++Xw) { *Xw = *X; }
                X -= xd; prev_n = ss + (int)stp;
            }
            ss += stp;
        }

        //Dither
        if (dither>FLT_EPSILON)
        {
            Xw -= L;
            for (size_t l=0u; l<L; l+=2u)
            {
                state = state*6364136223846793005u + inc;
                xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
                rot = state >> 59u;
                r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                u1 = ldexp((float)r,-32);
                state = state*6364136223846793005u + inc;
                xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
                rot = state >> 59u;
                r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                u2 = ldexp((float)r,-32);
                R = dither * sqrtf(-2.0f*logf(1.0f-u1));
                *Xw++ += R * cosf(M_2PI*u2);
                if (l+1u<L) { *Xw++ += R * sinf(M_2PI*u2); }
            }
        }

        //Zero DC (subtract mean)
        if (dc0)
        {
            float mn = 0.0f;
            for (size_t l=L; l>0u; --l) { mn += *--Xw; }
            mn /= (float)L;
            for (size_t l=L; l>0u; --l, ++Xw) { *Xw -= mn; }
        }

        //Raw energy
        if (use_energy && raw_energy)
        {
            Xw -= L; rawe = 0.0f;
            for (size_t l=L; l>0u; --l, ++Xw) { rawe += *Xw * *Xw; }
            if (rawe<rawe_floor) { rawe = rawe_floor; }
        }

        //Preemph
        if (L<2u || preemph<FLT_EPSILON) { Xw -= L; }
        else
        {
            --Xw;
            for (size_t l=L-1u; l>0u; --l, --Xw) { *Xw -= preemph * *(Xw-1); }
            *Xw -= preemph * *Xw;
        }
        
        //Window
        for (size_t l=L; l>0u; --l, ++Xw, ++win) { *Xw *= *win; }
        Xw -=L; win -=L;
        
        //Raw energy
        if (use_energy && !raw_energy)
        {
            rawe = 0.0f;
            for (size_t l=L; l>0u; --l, ++Xw) { rawe += *Xw * *Xw; }
            if (rawe<rawe_floor) { rawe = rawe_floor; }
            Xw -= L;
        }

        //FFT
        fftwf_execute(fft_plan);
        
        //Power (from fftw half-complex format)
        for (size_t f=F; f>0u; --f, ++Yw, ++Yf) { *Yf = *Yw * *Yw; }
        Yf -= 2u;
        for (size_t f=F-2u; f>0u; --f, ++Yw, --Yf) { *Yf += *Yw * *Yw; }
        Yw -= nfft;

        //Transform to mel-bank output (and apply floor if log)
        for (size_t b=B; b>0u; --b, ++beg_f, ++num_f, ++Xd)
        {
            float sm = 0.0f;
            Yf += *beg_f;
            for (size_t f=*num_f; f>0u; --f) { sm += *Yf++ * *F2B++; }
            *Xd = (sm<FLT_EPSILON) ? logf(FLT_EPSILON) : logf(sm);
            Yf -= *beg_f + *num_f;
        }
        beg_f -= B; num_f -= B; F2B -= nf2b; Xd -= B;

        //DCT
        fftwf_execute(dct_plan);

        //Lifter
        *Y = (use_energy) ? logf(rawe) : *lift * *Yd;
        ++lift; ++Yd; ++Y;
        for (size_t c=C-2u; c>0u; --c, ++lift, ++Yd, ++Y) { *Y = *lift * *Yd; }
        lift -= C-1u; Yd -= C-1u; ++Y;

        //This reproduces a bug in Kaldi (last 2 bins are equal)
        *(Y-1) = *(Y-2);
    }

    //Subtract means from Y
    if (mn0)
    {
        float *mns;
        if (!(mns=(float *)calloc(B,sizeof(float)))) { fprintf(stderr,"error in kaldi_mfcc_s: problem with calloc. "); perror("calloc"); return 1; }
        for (size_t w=W; w>0u; --w, mns-=B)
        {
            for (size_t b=B; b>0u; --b, ++mns) { *mns += *--Y; }
        }
        for (size_t b=B; b>0u; --b, ++mns) { *mns /= (float)W; }
        for (size_t w=W; w>0u; --w, mns+=B)
        {
            for (size_t b=B; b>0u; --b, ++Y) { *Y -= *--mns; }
        }
        mns -= B; Y -= B*W;
        free(mns);
    }
    
    //Free
    free(win); free(Yf); free(mels); free(F2B); free(beg_f); free(num_f); free(lift);
    fftwf_destroy_plan(fft_plan); fftwf_free(Xw); fftwf_free(Yw);
    fftwf_destroy_plan(dct_plan); fftwf_free(Xd); fftwf_free(Yd);

    return 0;
}


int kaldi_mfcc_d (double *Y, double *X, const size_t N, const double sr, const double frame_length, const double frame_shift, const int snip_edges, const double dither, const int dc0, const int raw_energy, const double preemph, const char win_type[], const double lof, const double hif, const size_t B, const size_t C, const double Q, const int use_energy, const int mn0)
{
    if (N<1u) { fprintf(stderr,"error in kaldi_mfcc_d: N (nsamps in signal) must be positive\n"); return 1; }
    if (sr<DBL_EPSILON) { fprintf(stderr,"error in kaldi_mfcc_d: sr must be positive\n"); return 1; }
    if (frame_length<DBL_EPSILON) { fprintf(stderr,"error in kaldi_mfcc_d: frame_length must be positive\n"); return 1; }
    if (frame_shift<DBL_EPSILON) { fprintf(stderr,"error in kaldi_mfcc_d: frame_shift must be positive\n"); return 1; }
    if (dither<0.0) { fprintf(stderr,"error in kaldi_mfcc_d: dither must be nonnegative\n"); return 1; }
    if (preemph<0.0) { fprintf(stderr,"error in kaldi_mfcc_d: preemph coeff must be nonnegative\n"); return 1; }
    if (lof<0.0) { fprintf(stderr,"error in kaldi_mfcc_d: lof (low-freq cutoff) must be nonnegative\n"); return 1; }
    if (hif<=lof) { fprintf(stderr,"error in kaldi_mfcc_d: hif (high-freq cutoff) must be > lof\n"); return 1; }
    if (B<1u) { fprintf(stderr,"error in kaldi_mfcc_d: B (num mel bins) must be positive\n"); return 1; }
    if (C<1u) { fprintf(stderr,"error in kaldi_mfcc_d: C (num cepstral coeffs) must be positive\n"); return 1; }
    if (C>B) { fprintf(stderr,"error in kaldi_mfcc_d: C (num cepstral coeffs) must be <= B (num mel bins)\n"); return 1; }
    if (Q<0.0) { fprintf(stderr,"error in kaldi_mfcc_d: Q (lifter coeff) must be nonnegative\n"); return 1; }
    
    //Set L (frame_length in samps)
    const size_t L = (size_t)(sr*frame_length/1000.0);
    if (L<1u) { fprintf(stderr,"error in kaldi_mfcc_d: frame_length must be greater than 1 sample\n"); return 1; }
    if (snip_edges && L>N) { fprintf(stderr,"error in kaldi_mfcc_d: frame length must be < signal length if snip_edges\n"); return 1; }

    //Set stp (frame_shift in samps)
    const size_t stp = (size_t)(sr*frame_shift/1000.0);
    if (stp<1u) { fprintf(stderr,"error in kaldi_mfcc_d: frame_shift must be greater than 1 sample\n"); return 1; }

    //Set W (number of frames or windows)
    const size_t W = (snip_edges) ? 1u+(N-L)/stp : (N+stp/2u)/stp;
    if (W==0u) { return 0; }

    //Initialize framing
    const size_t Lpre = L/2u;                   //nsamps before center samp
    int ss = (int)(stp/2u) - (int)Lpre;         //start-samp of current frame
    int n, prev_n = 0;                          //current/prev samps in X
    const int xd = (int)L - (int)stp;           //a fixed increment after each frame for speed below

    //Initialize dither (this is a direct randn generator using method of PCG library)
    const double FLT_EPS = (double)FLT_EPSILON;
    const double M_2PI = 2.0*M_PI;
    double u1, u2, R;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;
    if (dither>FLT_EPS)
    {
        //Init random num generator
        if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in kaldi_mfcc_d: problem with timespec_get.\n"); perror("timespec_get"); return 1; }
        state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;
    }

    //Initialize raw_energy
    const double rawe_floor = FLT_EPS;
    double rawe = 0.0;

    //Get win (window vec of length L)
    double *win;
    if (!(win=(double *)malloc(L*sizeof(double)))) { fprintf(stderr,"error in kaldi_mfcc_d: problem with malloc. "); perror("malloc"); return 1; }
    if (strncmp(win_type,"rectangular",11u)==0)
    {
        for (size_t l=0u; l<L; ++l) { win[l] = 1.0; }
    }
    else
    {
        const double a = 2.0*M_PI/(double)(L-1u);
        if (strncmp(win_type,"blackman",8u)==0)
        {
            const double coeff = 0.42, coeff1 = 1.0 - coeff; //blackman_coeff
            for (size_t l=0u; l<L; ++l) { win[l] = coeff - 0.5*cos((double)l*a) + coeff1*cos((double)(2u*l)*a); }
        }
        else if (strncmp(win_type,"hamming",7u)==0)
        {
            for (size_t l=0u; l<L; ++l) { win[l] = 0.54 - 0.46*cos((double)l*a); }
        }
        else if (strncmp(win_type,"hann",4u)==0)
        {
            for (size_t l=0u; l<L; ++l) { win[l] = 0.5 - 0.5*cos((double)l*a); }
        }
        else if (strncmp(win_type,"povey",5u)==0)
        {
            for (size_t l=0u; l<L; ++l) { win[l] = pow(0.5-0.5*cos((double)l*a),0.85); }
        }
        else
        {
            fprintf(stderr,"error in kaldi_mfcc_d: window type must be rectangular, blackman, hamming, hann or povey (lower-cased)\n"); return 1;
        }
    }

    //Set nfft (FFT transform length), which is next pow2 of L
    size_t nfft = 1u;
    while (nfft<L) { nfft *= 2u; }

    //Set F (num non-negative FFT freqs)
    const size_t F = nfft/2u + 1u;
    
    //Initialize FFT (using lib fftw3)
    double *Xw, *Yw;
    Xw = (double *)fftw_malloc(nfft*sizeof(double));
    Yw = (double *)fftw_malloc(nfft*sizeof(double));
    fftw_plan fft_plan = fftw_plan_r2r_1d((int)nfft,Xw,Yw,FFTW_R2HC,FFTW_ESTIMATE);
    if (!fft_plan) { fprintf(stderr,"error in kaldi_mfcc_d: problem creating fftw plan"); return 1; }
    for (size_t nf=nfft; nf>0u; --nf, ++Xw) { *Xw = 0.0; }
    Xw -= nfft;

    //Initialize Yf (initial output with power at F STFT freqs)
    double *Yf;
    if (!(Yf=(double *)malloc(F*sizeof(double)))) { fprintf(stderr,"error in kaldi_mfcc_d: problem with malloc. "); perror("malloc"); return 1; }

    //Initialize Hz-to-mel transfrom matrix (F2B)
    const double finc = sr/(double)nfft;                    //freq increment in Hz for FFT freqs
    const double lomel = 1127.0 * log(1.0+lof/700.0);       //low-mel cutoff
    const double himel = 1127.0 * log(1.0+hif/700.0);       //high-mel cutoff
    const double dmel = (himel-lomel) / (double)(B+1u);     //controls spacing on mel scale
    double lmel, cmel, rmel;                                //current left, center, right mels
    double *mels;                                           //maps STFT freqs to mels
    double *F2B;                                            //transform matrix for STFT power in F freqs to B mel bins
    size_t *beg_f, *num_f, nf2b = 0u;                       //indices and size of F2B as sparse matrix
    mels = (double *)malloc(F*sizeof(double));
    F2B = (double *)malloc(B*(F/2u)*sizeof(double));
    beg_f = (size_t *)malloc(B*sizeof(size_t));
    num_f = (size_t *)malloc(B*sizeof(size_t));
    if (!mels) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!F2B) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!beg_f) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!num_f) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    for (size_t f=0; f<F; ++f) { mels[f] = 1127.0 * log(1.0+(double)f*finc/700.0); }
    lmel = lomel; cmel = lmel + dmel; rmel = cmel + dmel;
    for (size_t b=B; b>0u; --b)
    {
        size_t f = 0u;
        while (*mels<lmel && f<F) { ++mels; ++f; }
        *beg_f = f;
        while (*mels<cmel && f<F) { *F2B++ = (*mels-lmel)/dmel; ++mels; ++f; }
        while (*mels<rmel && f<F) { *F2B++ = (rmel-*mels)/dmel; ++mels; ++f; }
        *num_f = f - *beg_f++; nf2b += *num_f++;
        mels -= f;
        lmel = cmel; cmel = rmel;
        rmel = (b==2u) ? himel : rmel + dmel;
    }
	F2B -= nf2b; beg_f -= B; num_f -= B;

    //Initialize DCT
    //Kaldi uses a dct_matrix, that is B x B, and then resized to nceps x B.
    double *Xd, *Yd;
    Xd = (double *)fftw_malloc(B*sizeof(double));
    Yd = (double *)fftw_malloc(B*sizeof(double));
    fftw_plan dct_plan = fftw_plan_r2r_1d((int)B,Xd,Yd,FFTW_REDFT10,FFTW_ESTIMATE);
    if (!dct_plan) { fprintf(stderr,"error in kaldi_mfcc_d: problem creating fftw plan"); return 1; }
    for (size_t b=B; b>0u; --b, ++Xd) { *Xd = 0.0; }
    Xd -= B;

    //Initialize lifter (include DCT scaling)
    const double sc = 1.0/sqrt((double)(2u*B));
    double *lift;
    if (!(lift=(double *)malloc(C*sizeof(double)))) { fprintf(stderr,"error in kaldi_mfcc_d: problem with malloc. "); perror("malloc"); return 1; }
    if (Q>FLT_EPS)
    {
        for (size_t c=0u; c<C; ++c, ++lift) { *lift = 1.0 + 0.5*Q*sin(M_PI*(double)c/Q); }
    }
    else { for (size_t c=0u; c<C; ++c, ++lift) { *lift = 1.0; } }
    for (size_t c=1u; c<C; ++c) { *--lift *= sc; }
    *--lift *= 0.5/sqrt((double)B);      //proper DC scaling

    //Process each of W frames
    for (size_t w=W; w>0u; --w)
    {
        //Copy one frame of X into Xw
        if (snip_edges)
        {
            for (size_t l=L; l>0u; --l, ++X, ++Xw) { *Xw = *X; }
            X -= xd;
        }
        else
        {
            if (ss<0 || ss>(int)N-(int)L)
            {
                for (int s=ss; s<ss+(int)L; ++s, ++Xw)
                {
                    n = s; //This ensures extrapolation by signal reversal to any length
                    while (n<0 || n>=(int)N) { n = (n<0) ? -n-1 : (n<(int)N) ? n : 2*(int)N-1-n; }
                    X += n - prev_n; prev_n = n;
                    *Xw = *X;
                }
            }
            else
            {
                X += ss - prev_n;
                for (size_t l=L; l>0u; --l, ++X, ++Xw) { *Xw = *X; }
                X -= xd; prev_n = ss + (int)stp;
            }
            ss += stp;
        }

        //Dither
        if (dither>FLT_EPS)
        {
            Xw -= L;
            for (size_t l=0u; l<L; l+=2u)
            {
                state = state*6364136223846793005u + inc;
                xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
                rot = state >> 59u;
                r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                u1 = ldexp((double)r,-32);
                state = state*6364136223846793005u + inc;
                xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
                rot = state >> 59u;
                r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
                u2 = ldexp((double)r,-32);
                R = dither * sqrt(-2.0*log(1.0-u1));
                *Xw++ += R * cos(M_2PI*u2);
                if (l+1u<L) { *Xw++ += R * sin(M_2PI*u2); }
            }
        }

        //Raw energy
        if (use_energy && raw_energy)
        {
            Xw -= L; rawe = 0.0;
            for (size_t l=L; l>0u; --l, ++Xw) { rawe += *Xw * *Xw; }
            if (rawe<rawe_floor) { rawe = rawe_floor; }
        }

        //Zero DC (subtract mean)
        if (dc0)
        {
            double mn = 0.0;
            for (size_t l=L; l>0u; --l) { mn += *--Xw; }
            mn /= (double)L;
            for (size_t l=L; l>0u; --l, ++Xw) { *Xw -= mn; }
        }

        //Preemph
        if (L<2u || preemph<FLT_EPS) { Xw -= L; }
        else
        {
            --Xw;
            for (size_t l=L-1u; l>0u; --l, --Xw) { *Xw -= preemph * *(Xw-1); }
            *Xw -= preemph * *Xw;
        }
        
        //Window
        for (size_t l=L; l>0u; --l, ++Xw, ++win) { *Xw *= *win; }
        Xw -= L; win -= L;
        
        //Raw energy
        if (use_energy && !raw_energy)
        {
            rawe = 0.0;
            for (size_t l=L; l>0u; --l, ++Xw) { rawe += *Xw * *Xw; }
            if (rawe<rawe_floor) { rawe = rawe_floor; }
            Xw -= L;
        }

        //FFT
        fftw_execute(fft_plan);
        
        //Power (from fftw half-complex format)
        for (size_t f=F; f>0u; --f, ++Yw, ++Yf) { *Yf = *Yw * *Yw; }
        Yf -= 2u;
        for (size_t f=F-2u; f>0u; --f, ++Yw, --Yf) { *Yf += *Yw * *Yw; }
        Yw -= nfft;

        //Transform to mel-bank output (and apply floor if log)
        for (size_t b=B; b>0u; --b, ++beg_f, ++num_f, ++Xd)
        {
            double sm = 0.0;
            Yf += *beg_f;
            for (size_t f=*num_f; f>0u; --f) { sm += *Yf++ * *F2B++; }
            *Xd = (sm<FLT_EPS) ? log(FLT_EPS) : log(sm);
            Yf -= *beg_f + *num_f;
        }
        beg_f -= B; num_f -= B; F2B -= nf2b; Xd -= B;

        //DCT
        fftw_execute(dct_plan);

        //Lifter
        *Y = (use_energy) ? log(rawe) : *lift * *Yd;
        ++lift; ++Yd; ++Y;
        for (size_t c=C-2u; c>0u; --c, ++lift, ++Yd, ++Y) { *Y = *lift * *Yd; }
        lift -= C-1u; Yd -= C-1u; ++Y;

        //This reproduces a bug in Kaldi (last 2 bins are equal)
        *(Y-1) = *(Y-2);
    }

    //Subtract means from Y
    if (mn0)
    {
        double *mns;
        if (!(mns=(double *)calloc(B,sizeof(double)))) { fprintf(stderr,"error in kaldi_plp_d: problem with calloc. "); perror("calloc"); return 1; }
        for (size_t w=W; w>0u; --w, mns-=B)
        {
            for (size_t b=B; b>0u; --b, ++mns) { *mns += *--Y; }
        }
        for (size_t b=B; b>0u; --b, ++mns) { *mns /= (double)W; }
        for (size_t w=W; w>0u; --w, mns+=B)
        {
            for (size_t b=B; b>0u; --b, ++Y) { *Y -= *--mns; }
        }
        mns -= B; Y -= B*W;
        free(mns);
    }
    
    //Free
    free(win); free(Yf); free(mels); free(F2B); free(beg_f); free(num_f); free(lift);
    fftw_destroy_plan(fft_plan); fftw_free(Xw); fftw_free(Yw);
    fftw_destroy_plan(dct_plan); fftw_free(Xd); fftw_free(Yd);

    return 0;
}


#ifdef __cplusplus
}
}
#endif
