//@author Erik Edwards
//@date 2018-present
//@license BSD 3-clause


#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <valarray>
#include <unordered_map>
#include <argtable2.h>
#include "../util/cmli.hpp"
#include <cfloat>
#include "kaldi_spectrogram.c"

#ifdef I
#undef I
#endif


int main(int argc, char *argv[])
{
    using namespace std;


    //Declarations
    int ret = 0;
    const string errstr = ": \033[1;31merror:\033[0m ";
    const string warstr = ": \033[1;35mwarning:\033[0m ";
    const string progstr(__FILE__,string(__FILE__).find_last_of("/")+1,strlen(__FILE__)-string(__FILE__).find_last_of("/")-5);
    const valarray<size_t> oktypes = {1u,2u};
    const size_t I = 1u, O = 1u;
    ifstream ifs1; ofstream ofs1;
    int8_t stdi1, stdo1, wo1;
    ioinfo i1, o1;
    size_t W, nfft, F, L, stp;
    double d, p, sr, fl, shft;
    int snipe, dc0, rawe, mn0;
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
    descr += "Use -r (--srate) to give the sample rate [default=16000].\n";
    descr += "\n";
    descr += "Use -t (--frame_length) to give the frame length in ms [default=25].\n";
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
    descr += "Use -d (--dither) to give the dither weight [default=0.1].\n";
    descr += "Set this to 0 to turn off dithering.\n";
    descr += "\n";
    descr += "Include -z (--zero_dc) to subtract the mean from each frame [default=false].\n";
    descr += "This is applied just after dither, before preemph [usually recommended].\n";
    descr += "\n";
    descr += "Include -y (--raw_energy) to compute energy before preemph [default=false].\n";
    descr += "This is computed just after zero_dc (before preemph and window) for true,\n";
    descr += "but just before FFT (after preemph and window) for false.\n";
    descr += "\n";
    descr += "Use -p (--preemph) to give the preemphasis.\n";
    descr += "\n";
    descr += "Include -m (--zero_mean) to subtract the means from Y [default=false].\n";
    descr += "This is takes F means and subtracts just before output [not usually recommended].\n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ kaldi_spectrogram X -o Y \n";
    descr += "$ kaldi_spectrogram -e X > Y \n";
    descr += "$ cat X | kaldi_spectrogram -e - > Y \n";
    descr += "$ kaldi_spectrogram -r8000 -d0 -w'hamming' -e -z X -o Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X1,X2)");
    struct arg_dbl   *a_sr = arg_dbln("r","srate","<dbl>",0,1,"sample rate in Hz [default=16000.0]");
    struct arg_dbl   *a_fl = arg_dbln("t","frame_length","<dbl>",0,1,"length in ms of each frame [default=25]");
    struct arg_dbl  *a_stp = arg_dbln("s","frame_shift","<dbl>",0,1,"step in ms between each frame [default=10]");
    struct arg_lit  *a_sne = arg_litn("e","snip_edges",0,1,"include to snip edges [default=false]");
    struct arg_dbl    *a_d = arg_dbln("d","dither","<dbl>",0,1,"dither weight [default=0.1]");
    struct arg_lit  *a_dc0 = arg_litn("z","zero_dc",0,1,"include to zero the mean of each frame [default=false]");
    struct arg_lit   *a_re = arg_litn("y","raw_energy",0,1,"include to compute energy before preemph [default=false]");
    struct arg_dbl    *a_p = arg_dbln("p","preemph","<dbl>",0,1,"preemphasis coefficient in [0 1] [default=0.97]");
    struct arg_str   *a_wt = arg_strn("w","wintype","<str>",0,1,"window type [default='povey']");
    struct arg_lit  *a_mn0 = arg_litn("m","zero_mean",0,1,"include to zero the means of each feat in Y [default=false]");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_sr, a_fl, a_stp, a_sne, a_d, a_dc0, a_re, a_p, a_wt, a_mn0, a_fo, a_help, a_end};
    if (arg_nullcheck(argtable)!=0) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating argtable" << endl; return 1; }
    nerrs = arg_parse(argc, argv, argtable);
    if (a_help->count>0)
    {
        cout << "Usage: " << progstr; arg_print_syntax(stdout, argtable, "\n");
        cout << endl; arg_print_glossary(stdout, argtable, "  %-25s %s\n");
        cout << endl << descr; return 1;
    }
    if (nerrs>0) { arg_print_errors(stderr,a_end,(progstr+": "+to_string(__LINE__)+errstr).c_str()); return 1; }


    //Check stdin
    stdi1 = (a_fi->count==0 || strlen(a_fi->filename[0])==0u || strcmp(a_fi->filename[0],"-")==0);
    if (stdi1>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0u || strcmp(a_fo->filename[0],"-")==0); }
    else { stdo1 = (!isatty(fileno(stdout))); }
    wo1 = (stdo1 || a_fo->count>0);


    //Open input
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file" << endl; return 1; }


    //Read input header
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1u]) ? "}" : ","); }
        cerr << endl; return 1;
    }


    //Get options

    //Get sr
    sr = (a_sr->count>0) ? a_sr->dval[0] : 16000.0;
    if (sr<DBL_EPSILON) { cerr << progstr+": " << __LINE__ << errstr << "srate (sample rate) must be nonnegative" << endl; return 1; }

    //Get fl
    fl = (a_fl->count>0) ? a_fl->dval[0] : 25.0;
    if (fl<DBL_EPSILON) { cerr << progstr+": " << __LINE__ << errstr << "frame length must be positive" << endl; return 1; }

    //Get shft
    shft = (a_stp->count>0) ? a_stp->dval[0] : 10.0;
    if (shft<DBL_EPSILON) { cerr << progstr+": " << __LINE__ << errstr << "frame shift must be positive" << endl; return 1; }

    //Get snip_edges
    snipe = (a_sne->count>0);

    //Get d
    d = (a_d->count>0) ? a_d->dval[0] : 0.1;
    if (d<0.0) { cerr << progstr+": " << __LINE__ << errstr << "d must be nonnegative" << endl; return 1; }

    //Get dc0
    dc0 = (a_dc0->count>0);

    //Get raw_energy
    rawe = (a_re->count>0);

    //Get p
    p = (a_p->count>0) ? a_p->dval[0] : 0.97;
    if (p<0.0 || p>1.0) { cerr << progstr+": " << __LINE__ << errstr << "p must be in [0.0 1.0]" << endl; return 1; }

    //Get wintype
    if (a_wt->count==0) { wintype = "povey"; }
    else
    {
    	try { wintype = string(a_wt->sval[0]); }
    	catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem getting string for window type" << endl; return 1; }
    }
    for (string::size_type c=0u; c<wintype.size(); ++c) { wintype[c] = char(tolower(wintype[c])); }

    //Get mn0
    mn0 = (a_mn0->count>0);


    //Checks
    if (!i1.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a vector" << endl; return 1; }
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }


    //Set output header info
    L = size_t(sr*fl/1000.0);
    stp = size_t(sr*shft/1000.0);
    nfft = 1u;
    while (nfft<L) { nfft *= 2u; }
    F = nfft/2u + 1u;
    W = (snipe) ? 1u+(i1.N()-L)/stp : (i1.N()+stp/2u)/stp;
    o1.F = i1.F; o1.T = i1.T;
    o1.R = (i1.isrowmajor()) ? W : F;
    o1.C = (i1.isrowmajor()) ? F : W;
    o1.S = i1.S; o1.H = i1.H;


    //Open output
    if (wo1)
    {
        if (stdo1) { ofs1.copyfmt(cout); ofs1.basic_ios<char>::rdbuf(cout.rdbuf()); } else { ofs1.open(a_fo->filename[0]); }
        if (!ofs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening output file 1" << endl; return 1; }
    }


    //Write output header
    if (wo1 && !write_output_header(ofs1,o1)) { cerr << progstr+": " << __LINE__ << errstr << "problem writing header for output file 1" << endl; return 1; }


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
        if (codee::kaldi_spectrogram_s(Y,X,i1.N(),float(sr),float(fl),float(shft),snipe,float(d),dc0,rawe,float(p),wintype.c_str(),mn0))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Y;
    }
    else if (o1.T==2u)
    {
        double *X, *Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
        try { Y = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
        if (codee::kaldi_spectrogram_d(Y,X,i1.N(),double(sr),double(fl),double(shft),snipe,double(d),dc0,rawe,double(p),wintype.c_str(),mn0))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] Y;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    

    //Exit
    return ret;
}
