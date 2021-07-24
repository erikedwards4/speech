//Kaldi-format "fbank" features
//STFT (short-term Fourier transform) of univariate time series X,
//outputing power on a mel frequency scale in Y.

//This takes a univariate time series (vector) X of length N, and
//outputs the power in each of B mel-frequency bands at each of W windows.
//If Y is row-major, then it has size W x B.
//If Y is col-major, then it has size B x W.
//where B is the number of frequency bands, or filter bank members,
//also referred to as the number of mel bins.

//This follows Kaldi and torchaudio conventions for compatibility.

//The following params and boolean options are included:
//N:            size_t  length of input signal X in samps
//fs:           float   sample rate in Hz
//frame_length: float   length of each frame (window) in msec (often 25 ms)
//frame_shift:  float   step size between each frame center in msec (often 10 ms)
//snip_edges:   bool    controls style of framing w.r.t. edges of X
//raw_energy:   bool    use raw energy of each frame instead of DC^2 from FFT
//dither:       float   dithering weight (set to 0.0 for no dither)
//dc0:          bool    to subtract mean from each frame after dither
//preemph:      float   preemph coeff (set to 0.0 for no preemph)
//win_type:     string  window type in {rectangular,blackman,hamming,hann,povey} 
//amp:          bool    use magnitude (amplitude) rather than power after FFT
//B:            size_t  num mel bins (typical default is 23)
//lof:          float   low freq cutoff for mel bins (typical default is 20 Hz)
//hif:          float   high freq cutoff for mel bins (typical default is Nyquist)
//lg:           bool    take log after getting mel-bank power
//mn0:          bool    to subtract means of B feats in Y before output (not usually recommended)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

#ifndef M_PI
   #define M_PI 3.141592653589793238462643383279502884
#endif

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int kaldi_fbank_s (float *Y, float *X, const size_t N, const float fs, const float frame_length, const float frame_shift, const int snip_edges, const int raw_energy, const float dither, const int dc0, const float preemph, const char win_type[], const int amp, const size_t B, const float lof, const float hif, const int lg, const int mn0);
int kaldi_fbank_d (double *Y, double *X, const size_t N, const double fs, const double frame_length, const double frame_shift, const int snip_edges, const int raw_energy, const double dither, const int dc0, const double preemph, const char win_type[], const int amp, const size_t B, const double lof, const double hif, const int lg, const int mn0);


int kaldi_fbank_s (float *Y, float *X, const size_t N, const float fs, const float frame_length, const float frame_shift, const int snip_edges, const int raw_energy, const float dither, const int dc0, const float preemph, const char win_type[], const int amp, const size_t B, const float lof, const float hif, const int lg, const int mn0)
{
    if (N<1u) { fprintf(stderr,"error in kaldi_fbank_s: N (nsamps in signal) must be positive\n"); return 1; }
    if (fs<FLT_EPSILON) { fprintf(stderr,"error in kaldi_fbank_s: fs must be positive\n"); return 1; }
    if (frame_length<FLT_EPSILON) { fprintf(stderr,"error in kaldi_fbank_s: frame_length must be positive\n"); return 1; }
    if (frame_shift<FLT_EPSILON) { fprintf(stderr,"error in kaldi_fbank_s: frame_shift must be positive\n"); return 1; }
    if (dither<0.0f) { fprintf(stderr,"error in kaldi_fbank_s: dither must be nonnegative\n"); return 1; }
    if (preemph<0.0f) { fprintf(stderr,"error in kaldi_fbank_s: preemph coeff must be nonnegative\n"); return 1; }
    if (lof<0.0f) { fprintf(stderr,"error in kaldi_fbank_s: lof (low-freq cutoff) must be nonnegative\n"); return 1; }
    if (hif<=lof) { fprintf(stderr,"error in kaldi_fbank_s: hif (high-freq cutoff) must be > lof\n"); return 1; }
    if (B<1u) { fprintf(stderr,"error in kaldi_fbank_s: B (num mel bins) must be positive\n"); return 1; }

    //Set L (frame_length in samps)
    const size_t L = (size_t)(fs*frame_length/1000.0f);
    if (L<1u) { fprintf(stderr,"error in kaldi_fbank_s: frame_length must be greater than 1 sample\n"); return 1; }
    if (snip_edges && L>N) { fprintf(stderr,"error in kaldi_fbank_s: frame length must be < signal length if snip_edges\n"); return 1; }

    //Set stp (frame_shift in samps)
    const size_t stp = (size_t)(fs*frame_shift/1000.0f);
    if (stp<1u) { fprintf(stderr,"error in kaldi_fbank_s: frame_shift must be greater than 1 sample\n"); return 1; }

    //Set W (number of frames or windows)
    const size_t W = (snip_edges) ? 1u+(N-L)/stp : (N+stp/2u)/stp;
    if (W==0u) { return 0; }

    //Initialize framing
    const size_t Lpre = L/2u;                   //nsamps before center samp
    int ss = (int)(stp/2u) - (int)Lpre;         //start-samp of current frame
    int n, prev_n = 0;                          //current/prev samps in X
    const int xd = (int)L - (int)stp;           //a fixed increment after each frame for speed below

    //Initialize raw_energy
    const float rawe_floor = FLT_EPSILON;
    float rawe = 0.0f;

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
        if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in kaldi_fbank_s: problem with timespec_get.\n"); perror("timespec_get"); return 1; }
        state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;
    }

    //Get win (window vec of length L)
    float *win;
    if (!(win=(float *)malloc(L*sizeof(float)))) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    if (strncmp(win_type,"rectangular",1u)==0)
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
            fprintf(stderr,"error in kaldi_fbank_s: window type must be rectangular, blackman, hamming, hann or povey (lower-cased)\n"); return 1;
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
    fftwf_plan plan = fftwf_plan_r2r_1d((int)nfft,Xw,Yw,FFTW_R2HC,FFTW_ESTIMATE);
    if (!plan) { fprintf(stderr,"error in kaldi_fbank_s: problem creating fftw plan"); return 1; }
    for (size_t nf=0u; nf<nfft; ++nf) { Xw[nf] = 0.0f; }

    //Initialize Hz-to-mel transfrom matrix (F2B)
    const size_t BF = B*F;
    const float finc = fs/(float)nfft;                          //freq increment in Hz for FFT freqs
    const float lomel = 2595.0f * log10f((1.0f+lof)/700.0f);    //low-mel cutoff
    const float himel = 2595.0f * log10f((1.0f+hif)/700.0f);    //high-mel cutoff
    const float dmel = (himel-lomel) / (float)(B+1u);           //controls spacing on mel scale
    float *mels;                                                //map STFT freqs to mels
    float *F2B;                                                 //transform matrix for STFT power in F freqs to B mel bins
    if (!(mels=(float *)malloc(F*sizeof(float)))) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F2B=(float *)malloc(BF*sizeof(float)))) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }
    for (size_t f=0; f<F; ++f) { mels[f] = 2595.0f * log10f((1.0f+((float)f*finc))/700.0f); }
    float lmel = lomel, cmel = lmel + dmel, rmel = cmel + dmel;
    for (size_t b=0u; b<B; ++b)
    {
        size_t f = 0u;
        while (*mels<lmel && f<F) { *F2B++ = 0.0f; ++mels; ++f; }
        while (*mels<cmel && f<F) { *F2B++ = (*mels-lmel)/dmel; ++mels; ++f; }
        while (*mels<rmel && f<F) { *F2B++ = (rmel-*mels)/dmel; ++mels; ++f; }
        mels -= f;
        while (f<F) { *F2B++ = 0.0f; ++f; }
        lmel = cmel; cmel = rmel;
        rmel = (b+2u==B) ? himel : rmel + dmel;
    }
	F2B -= BF;

    //Initialize Yf (initial output with power at F STFT freqs)
    float *Yf;
    if (!(Yf=(float *)malloc(F*sizeof(float)))) { fprintf(stderr,"error in kaldi_fbank_s: problem with malloc. "); perror("malloc"); return 1; }

    //Process each of W frames
    for (size_t w=0u; w<W; ++w)
    {
        //Copy one frame of X into Xw
        if (snip_edges)
        {
            for (size_t l=0u; l<L; ++l, ++X, ++Xw) { *Xw = *X; }
            X -= xd;
        }
        else
        {
            if (ss<0 || ss>(int)N-(int)L)
            {
                for (int s=ss; s<ss+(int)L; ++s, ++Xw)
                {
                    n = s; //This ensures extrapolation by signal reversal to any length
                    while (n<0 || n>=(int)N) { n = (n<0) ? -n : (n<(int)N) ? n : 2*(int)N-2-n; }
                    X += n - prev_n; prev_n = n;
                    *Xw = *X;
                }
            }
            else
            {
                X += ss - prev_n;
                for (size_t l=0u; l<L; ++l, ++X, ++Xw) { *Xw = *X; }
                X -= xd; prev_n = ss + (int)stp;
            }
            ss += stp;
        }

        //Raw energy
        if (raw_energy)
        {
            rawe = 0.0f;
            for (size_t l=0u; l<L; ++l) { rawe += Xw[l] * Xw[l]; }
            if (rawe<rawe_floor) { rawe = rawe_floor; }
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
            for (size_t l=0u; l<L; ++l) { mn += *--Xw; }
            mn /= (float)L;
            for (size_t l=0u; l<L; ++l, ++Xw) { *Xw -= mn; }
        }

        //Preemph
        if (L<2u || preemph<FLT_EPSILON) { Xw -= L; }
        else
        {
            --Xw;
            for (size_t l=0u; l<L-1u; ++l, --Xw) { *Xw -= preemph * *(Xw-1); }
            *Xw -= preemph * *Xw;
        }
        
        //Window
        for (size_t l=0u; l<L; ++l) { Xw[l] *= win[l]; }
        
        //FFT
        fftwf_execute(plan);
        
        //Power (from fftw half-complex format)
        *Yf = (raw_energy) ? rawe : *Yw**Yw + FLT_EPSILON;
        ++Yf; ++Yw;
        for (size_t f=1u; f<F; ++f, ++Yw, ++Yf) { *Yf = *Yw * *Yw; }
        Yf -= 2u;
        for (size_t f=1u; f<F-1u; ++f, ++Yw, --Yf) { *Yf += *Yw * *Yw; }
        Yw -= nfft;

        //Amplitude
        if (amp)
        {
            for (size_t f=0u; f<F; ++f) { Yf[f] = sqrtf(Yf[f]); }
        }

        //Transform to mel-bank output
        for (size_t b=0u; b<B; ++b, Yf-=F, ++Y)
        {
            float sm = 0.0f;
            for (size_t f=0u; f<F; ++f, ++Yf, ++F2B) { sm += *Yf * *F2B; }
            *Y = (lg) ? logf(sm+FLT_EPSILON) : sm;
        }
        F2B -= BF;

        //Transform to mel-bank output (slightly faster but harder to follow)
        // lmel = lomel; cmel = lmel + dmel; rmel = cmel + dmel;
        // for (size_t b=0u; b<B; ++b)
        // {
        //     size_t f = 0u; float sm = 0.0f;
        //     while (*mels<lmel && f<F) { ++mels; ++f; ++Yf; ++F2B; }
        //     while (*mels<rmel && f<F) { sm += *Yf * *F2B; ++mels; ++f; ++Yf; ++F2B; }
        //     *Y = (lg) ? logf(sm+FLT_EPSILON) : sm;
        //     mels -= f; Yf -= f; F2B += F-f;
        //     lmel = cmel; cmel = rmel;
        //     rmel = (b+2u==B) ? himel : rmel + dmel;
        // }
        // F2B -= BF;
    }

    //Subtract means from Y
    if (mn0)
    {
        float *mns;
        if (!(mns=(float *)calloc(B,sizeof(float)))) { fprintf(stderr,"error in kaldi_fbank_s: problem with calloc. "); perror("calloc"); return 1; }
        for (size_t w=0u; w<W; ++w, mns-=B)
        {
            for (size_t b=0u; b<B; ++b, ++mns) { *mns += *--Y; }
        }
        for (size_t b=0u; b<B; ++b, ++mns) { *mns /= (float)W; }
        for (size_t w=0u; w<W; ++w, mns+=B)
        {
            for (size_t b=0u; b<B; ++b, ++Y) { *Y -= *--mns; }
        }
        mns-=B; Y-=B*W;
        free(mns);
    }
    
    //Free
    free(win); free(mels); free(F2B); free(Yf);
    fftwf_destroy_plan(plan); fftwf_free(Xw); fftwf_free(Yw);

    return 0;
}


int kaldi_fbank_d (double *Y, double *X, const size_t N, const double fs, const double frame_length, const double frame_shift, const int snip_edges, const int raw_energy, const double dither, const int dc0, const double preemph, const char win_type[], const int amp, const size_t B, const double lof, const double hif, const int lg, const int mn0)
{
    if (N<1u) { fprintf(stderr,"error in kaldi_fbank_d: N (nsamps in signal) must be positive\n"); return 1; }
    if (fs<DBL_EPSILON) { fprintf(stderr,"error in kaldi_fbank_d: fs must be positive\n"); return 1; }
    if (frame_length<DBL_EPSILON) { fprintf(stderr,"error in kaldi_fbank_d: frame_length must be positive\n"); return 1; }
    if (frame_shift<DBL_EPSILON) { fprintf(stderr,"error in kaldi_fbank_d: frame_shift must be positive\n"); return 1; }
    if (dither<0.0) { fprintf(stderr,"error in kaldi_fbank_d: dither must be nonnegative\n"); return 1; }
    if (preemph<0.0) { fprintf(stderr,"error in kaldi_fbank_d: preemph coeff must be nonnegative\n"); return 1; }
    if (lof<0.0) { fprintf(stderr,"error in kaldi_fbank_d: lof (low-freq cutoff) must be nonnegative\n"); return 1; }
    if (hif<=lof) { fprintf(stderr,"error in kaldi_fbank_d: hif (high-freq cutoff) must be > lof\n"); return 1; }
    if (B<1u) { fprintf(stderr,"error in kaldi_fbank_d: B (num mel bins) must be positive\n"); return 1; }

    //Set L (frame_length in samps)
    const size_t L = (size_t)(fs*frame_length/1000.0);
    if (L<1u) { fprintf(stderr,"error in kaldi_fbank_d: frame_length must be greater than 1 sample\n"); return 1; }
    if (snip_edges && L>N) { fprintf(stderr,"error in kaldi_fbank_d: frame length must be < signal length if snip_edges\n"); return 1; }

    //Set stp (frame_shift in samps)
    const size_t stp = (size_t)(fs*frame_shift/1000.0);
    if (stp<1u) { fprintf(stderr,"error in kaldi_fbank_d: frame_shift must be greater than 1 sample\n"); return 1; }

    //Set W (number of frames or windows)
    const size_t W = (snip_edges) ? 1u+(N-L)/stp : (N+stp/2u)/stp;
    if (W==0u) { return 0; }

    //Initialize framing
    const size_t Lpre = L/2u;                   //nsamps before center samp
    int ss = (int)(stp/2u) - (int)Lpre;         //start-samp of current frame
    int n, prev_n = 0;                          //current/prev samps in X
    const int xd = (int)L - (int)stp;           //a fixed increment after each frame for speed below

    //Initialize raw_energy
    const double rawe_floor = (double)FLT_EPSILON;
    double rawe = 0.0;

    //Initialize dither (this is a direct randn generator using method of PCG library)
    const double M_2PI = 2.0*M_PI;
    double u1, u2, R;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;
    if (dither>DBL_EPSILON)
    {
        //Init random num generator
        if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in kaldi_fbank_d: problem with timespec_get.\n"); perror("timespec_get"); return 1; }
        state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;
    }

    //Get win (window vec of length L)
    double *win;
    if (!(win=(double *)malloc(L*sizeof(double)))) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    if (strncmp(win_type,"rectangular",1u)==0)
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
            fprintf(stderr,"error in kaldi_fbank_d: window type must be rectangular, blackman, hamming, hann or povey (lower-cased)\n"); return 1;
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
    fftw_plan plan = fftw_plan_r2r_1d((int)nfft,Xw,Yw,FFTW_R2HC,FFTW_ESTIMATE);
    if (!plan) { fprintf(stderr,"error in kaldi_fbank_d: problem creating fftw plan"); return 1; }
    for (size_t nf=0u; nf<nfft; ++nf) { Xw[nf] = 0.0; }

    //Initialize Hz-to-mel transfrom matrix (F2B)
    const size_t BF = B*F;
    const double finc = fs/(double)nfft;                     //freq increment in Hz for FFT freqs
    const double lomel = 2595.0 * log10((1.0+lof)/700.0);    //low-mel cutoff
    const double himel = 2595.0 * log10((1.0+hif)/700.0);    //high-mel cutoff
    const double dmel = (himel-lomel) / (double)(B+1u);      //controls spacing on mel scale
    double *mels;                                            //map STFT freqs to mels
    double *F2B;                                             //transform matrix for STFT power in F freqs to B mel bins
    if (!(mels=(double *)malloc(F*sizeof(double)))) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F2B=(double *)malloc(BF*sizeof(double)))) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }
    for (size_t f=0; f<F; ++f) { mels[f] = 2595.0 * log10((1.0+((double)f*finc))/700.0); }
    double lmel = lomel, cmel = lmel + dmel, rmel = cmel + dmel;
    for (size_t b=0u; b<B; ++b)
    {
        size_t f = 0u;
        while (*mels<lmel && f<F) { *F2B++ = 0.0; ++mels; ++f; }
        while (*mels<cmel && f<F) { *F2B++ = (*mels-lmel)/dmel; ++mels; ++f; }
        while (*mels<rmel && f<F) { *F2B++ = (rmel-*mels)/dmel; ++mels; ++f; }
        mels -= f;
        while (f<F) { *F2B++ = 0.0; ++f; }
        lmel = cmel; cmel = rmel;
        rmel = (b+2u==B) ? himel : rmel + dmel;
    }
	F2B -= BF;

    //Initialize Yf (initial output with power at F STFT freqs)
    double *Yf;
    if (!(Yf=(double *)malloc(F*sizeof(double)))) { fprintf(stderr,"error in kaldi_fbank_d: problem with malloc. "); perror("malloc"); return 1; }

    //Process each of W frames
    for (size_t w=0u; w<W; ++w)
    {
        //Copy one frame of X into Xw
        if (snip_edges)
        {
            for (size_t l=0u; l<L; ++l, ++X, ++Xw) { *Xw = *X; }
            X -= xd;
        }
        else
        {
            if (ss<0 || ss>(int)N-(int)L)
            {
                for (int s=ss; s<ss+(int)L; ++s, ++Xw)
                {
                    n = s; //This ensures extrapolation by signal reversal to any length
                    while (n<0 || n>=(int)N) { n = (n<0) ? -n : (n<(int)N) ? n : 2*(int)N-2-n; }
                    X += n - prev_n; prev_n = n;
                    *Xw = *X;
                }
            }
            else
            {
                X += ss - prev_n;
                for (size_t l=0u; l<L; ++l, ++X, ++Xw) { *Xw = *X; }
                X -= xd; prev_n = ss + (int)stp;
            }
            ss += stp;
        }

        //Raw energy
        if (raw_energy)
        {
            rawe = 0.0;
            for (size_t l=0u; l<L; ++l) { rawe += Xw[l] * Xw[l]; }
            if (rawe<rawe_floor) { rawe = rawe_floor; }
        }

        //Dither
        if (dither>(double)FLT_EPSILON)
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

        //Zero DC (subtract mean)
        if (dc0)
        {
            double mn = 0.0;
            for (size_t l=0u; l<L; ++l) { mn += *--Xw; }
            mn /= (double)L;
            for (size_t l=0u; l<L; ++l, ++Xw) { *Xw -= mn; }
        }

        //Preemph
        if (L<2u || preemph<(double)FLT_EPSILON) { Xw -= L; }
        else
        {
            --Xw;
            for (size_t l=0u; l<L-1u; ++l, --Xw) { *Xw -= preemph * *(Xw-1); }
            *Xw -= preemph * *Xw;
        }
        
        //Window
        for (size_t l=0u; l<L; ++l) { Xw[l] *= win[l]; }
        
        //FFT
        fftw_execute(plan);
        
        //Power (from fftw half-complex format)
        *Yf = (raw_energy) ? rawe : *Yw**Yw + (double)FLT_EPSILON;
        ++Yf; ++Yw;
        for (size_t f=1u; f<F; ++f, ++Yw, ++Yf) { *Yf = *Yw**Yw + (double)FLT_EPSILON; }
        Yf -= 2u;
        for (size_t f=1u; f<F-1u; ++f, ++Yw, --Yf) { *Yf += *Yw * *Yw; }
        Yw -= nfft;

        //Amplitude
        if (amp)
        {
            for (size_t f=0u; f<F; ++f) { Yf[f] = sqrt(Yf[f]); }
        }

        //Transform to mel-bank output
        for (size_t b=0u; b<B; ++b, Yf-=F, ++Y)
        {
            double sm = 0.0;
            for (size_t f=0u; f<F; ++f, ++Yf, ++F2B) { sm += *Yf * *F2B; }
            *Y = (lg) ? log(sm+(double)FLT_EPSILON) : sm;
        }
        F2B -= BF;

        //Transform to mel-bank output (slightly faster but harder to follow)
        // lmel = lomel; cmel = lmel + dmel; rmel = cmel + dmel;
        // for (size_t b=0u; b<B; ++b)
        // {
        //     size_t f = 0u; double sm = 0.0;
        //     while (*mels<lmel && f<F) { ++mels; ++f; ++Yf; ++F2B; }
        //     while (*mels<rmel && f<F) { sm += *Yf * *F2B; ++mels; ++f; ++Yf; ++F2B; }
        //     *Y = (lg) ? log(sm+DBL_EPSILON) : sm;
        //     mels -= f; Yf -= f; F2B += F-f;
        //     lmel = cmel; cmel = rmel;
        //     rmel = (b+2u==B) ? himel : rmel + dmel;
        // }
        // F2B -= BF;
    }

    //Subtract means from Y
    if (mn0)
    {
        double *mns;
        if (!(mns=(double *)calloc(B,sizeof(double)))) { fprintf(stderr,"error in kaldi_fbank_d: problem with calloc. "); perror("calloc"); return 1; }
        for (size_t w=0u; w<W; ++w, mns-=B)
        {
            for (size_t b=0u; b<B; ++b, ++mns) { *mns += *--Y; }
        }
        for (size_t b=0u; b<B; ++b, ++mns) { *mns /= (double)W; }
        for (size_t w=0u; w<W; ++w, mns+=B)
        {
            for (size_t b=0u; b<B; ++b, ++Y) { *Y -= *--mns; }
        }
        mns-=B; Y-=B*W;
        free(mns);
    }
    
    //Free
    free(win); free(mels); free(F2B); free(Yf);
    fftw_destroy_plan(plan); fftw_free(Xw); fftw_free(Yw);

    return 0;
}


#ifdef __cplusplus
}
}
#endif
