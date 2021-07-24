//Includes
#include <cfloat>
#include "kaldi_spectrogram.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
size_t W, nfft, F, L, stp;
double d, p, fs, fl, shft;
int snipe, rawe, dc0, mn0;
string wintype;

//Description
string descr;
descr += "Makes Kaldi spectrogram features for univariate, real-valued X.\n";
descr += "\n";
descr += "Each frame of X is windowed (element-wise multiplied) by a window;\n";
descr += "the FFT is done on each windowed frame; and the real-valued power\n";
descr += "at each positive FFT freq is returned.\n";
descr += "\n";
descr += "The output Y has size FxW or WxF, where F is nfft/2+1, \n";
descr += "and nfft is the next-pow-2 of L, \n";
descr += "and W is the number of frames (a.k.a. windows).\n";
descr += "\n";
descr += "Use -r (--fs) to give the sample rate [default=16000].\n";
descr += "\n";
descr += "Use -l (--frame_length) to give the frame length in ms [default=25].\n";
descr += "\n";
descr += "Use -s (--frame_shift) to give the frame shift in ms [default=10].\n";
descr += "\n";
descr += "Use -e (--snip_edges) to set snip_edges to true [default=false].\n";
descr += "This is a setting from HTK, Kaldi, Librosa, etc., which controls\n";
descr += "the placement of the first/last frames w.r.t. the start/end of X1.\n";
descr += "\n";
descr += "The number of output frames (W) is set as in Kaldi:\n";
descr += "If snip_edges=true:  W = 1u + (N-L)/stp   \n";
descr += "If snip_edges=false: W = (N+stp/2u) / stp \n";
descr += "\n";
descr += "If snip_edges=true, the first frame starts at samp 0,\n";
descr += "and the last frame fits entirely within the length of X.\n";
descr += "If snip_edges=false, the first frame is centered at samp stp/2,\n";
descr += "and the last frame can overlap the end of X.\n";
descr += "\n";
descr += "The following framing convention is used here:\n";
descr += "Samples from one frame are contiguous in memory, for row- and col-major.\n";
descr += "So, if Y is row-major, then it has size W x F; \n";
descr += "but if Y is col-major, then it has size F x W. \n";
descr += "\n";
descr += "Include -g (--raw_energy) to use raw energy [default=false].\n";
descr += "This is computed before dithering or windowing,\n";
descr += "and is substituted in place of the FFT DC term.\n";
descr += "\n";
descr += "Use -d (--dweight) to give the dither weight.\n";
descr += "\n";
descr += "Include -c (--zero_dc) to subtract the mean from each frame [default=false].\n";
descr += "This is applied just after dither, before the preemph.\n";
descr += "\n";
descr += "Use -p (--preemph) to give the preemphasis.\n";
descr += "\n";
descr += "Include -z (--zero_mean) to subtract the means from Y [default=false].\n";
descr += "This is takes F means and subtracts just before output [not usually recommended].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ kaldi_spectrogram X -o Y \n";
descr += "$ kaldi_spectrogram -e X > Y \n";
descr += "$ cat X | kaldi_spectrogram -e - > Y \n";
descr += "$ kaldi_spectrogram -r8000 -d0 -w'hamming' -e -z X -o Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X1,X2)");
struct arg_dbl   *a_sr = arg_dbln("r","fs","<dbl>",0,1,"sample rate in Hz [default=16000.0]");
struct arg_dbl   *a_fl = arg_dbln("l","frame_length","<dbl>",0,1,"length in ms of each frame [default=25]");
struct arg_dbl  *a_stp = arg_dbln("s","frame_shift","<dbl>",0,1,"step in ms between each frame [default=10]");
struct arg_lit  *a_sne = arg_litn("e","snip_edges",0,1,"include to snip edges [default=false]");
struct arg_lit   *a_re = arg_litn("y","raw_energy",0,1,"include to use raw energy [default=false]");
struct arg_dbl    *a_d = arg_dbln("d","dither","<dbl>",0,1,"dither coefficient (weight) [default=0.1]");
struct arg_lit  *a_dc0 = arg_litn("c","zero_dc",0,1,"include to zero the mean of each frame [default=false]");
struct arg_dbl    *a_p = arg_dbln("p","preemph","<dbl>",0,1,"preemphasis coefficient in [0 1] [default=0.97]");
struct arg_str   *a_wt = arg_strn("w","wintype","<str>",0,1,"window type [default='povey']");
struct arg_lit  *a_mn0 = arg_litn("z","zero_mean",0,1,"include to zero the means of each feat in Y [default=false]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get fs
fs = (a_sr->count>0) ? a_sr->dval[0] : 16000.0;
if (fs<DBL_EPSILON) { cerr << progstr+": " << __LINE__ << errstr << "fs (sample rate) must be nonnegative" << endl; return 1; }

//Get d
d = (a_d->count>0) ? a_d->dval[0] : 0.1;
if (d<0.0) { cerr << progstr+": " << __LINE__ << errstr << "d must be nonnegative" << endl; return 1; }

//Get p
p = (a_p->count>0) ? a_p->dval[0] : 0.97;
if (p<0.0 || p>1.0) { cerr << progstr+": " << __LINE__ << errstr << "p must be in [0.0 1.0]" << endl; return 1; }

//Get fl
fl = (a_fl->count>0) ? a_fl->dval[0] : 25.0;
if (fl<DBL_EPSILON) { cerr << progstr+": " << __LINE__ << errstr << "frame length must be positive" << endl; return 1; }

//Get shft
shft = (a_stp->count>0) ? a_stp->dval[0] : 10.0;
if (shft<DBL_EPSILON) { cerr << progstr+": " << __LINE__ << errstr << "frame shift must be positive" << endl; return 1; }

//Get snip_edges
snipe = (a_sne->count>0);

//Get raw_energy
rawe = (a_re->count>0);

//Get wintype
if (a_wt->count==0) { wintype = "povey"; }
else
{
	try { wintype = string(a_wt->sval[0]); }
	catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem getting string for window type" << endl; return 1; }
}
for (string::size_type c=0u; c<wintype.size(); ++c) { wintype[c] = char(tolower(wintype[c])); }

//Get dc0
dc0 = (a_dc0->count>0);

//Get mn0
mn0 = (a_mn0->count>0);

//Checks
if (!i1.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a vector" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }

//Set output header
L = size_t(fs*fl/1000.0);
stp = size_t(fs*shft/1000.0);
nfft = 1u;
while (nfft<L) { nfft *= 2u; }
F = nfft/2u + 1u;
W = (snipe) ? 1u+(i1.N()-L)/stp : (i1.N()+stp/2u)/stp;
o1.F = i1.F; o1.T = i1.T;
o1.R = (i1.isrowmajor()) ? W : F;
o1.C = (i1.isrowmajor()) ? F : W;
o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (o1.T==1u)
{
    float *X, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (codee::kaldi_spectrogram_s(Y,X,i1.N(),float(fs),float(fl),float(shft),snipe,rawe,float(d),dc0,float(p),wintype.c_str(),mn0))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Y;
}

//Finish
