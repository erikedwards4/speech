# speech

speech: functions for speech and voice processing  

================================================

This is a set of C functions, and associated command-line tools in C++,  
that implement speech and voice processing methods.  

The command-line programs are written in C++ with a consistent style and interface.  
The low-level functions themselves are written in C for fastest performance (like openBLAS).  

The C functions are meant for the developer; the C++ command-line tools are meant for the end-user.  

The interface to each C function is BLAS-like, meaning that one specifies the input and/or output dimensions,  
the matrix order as row-major or column-major, and so on.  

The C++ command-line programs are written in a consistent style that was developed for command-line tools in general.  
All of these command-line tools use argtable2 (http://argtable.sourceforge.net/) for parsing  
inputs and option flags. All of them allow -h (--help) as a flag to give description and usage info.  

Input/output is supported for NumPy tensors (https://numpy.org/)  
and several C++ tensor formats: Armadillo (http://arma.sourceforge.net/),  
ArrayFire (https://arrayfire.com/), and a minimal format for Eigen (http://eigen.tuxfamily.org/).  


## Dependencies
Requires argtable2, openBLAS, LAPACKE, FFTW.  
For Ubuntu, these are available by apt-get:  
```
sudo apt-get install libargtable2-0 libblas3 libopenblas-base liblapack3 liblapacke fftw3  
```

You must first install the util library:  
https://github.com/erikedwards4/util  
And install speech into the same parent directory as util.  
Preferably: /opt/codee/util and /opt/codee/speech  
For full examples and support functions, also install math, dsp and aud:  
https://github.com/erikedwards4/math  
https://github.com/erikedwards4/dsp  
https://github.com/erikedwards4/aud  



## Installation
```
cd /opt/codee  
git clone https://github.com/erikedwards4/speech  
cd /opt/codee/speech  
make  
```

Each C function can also be compiled and used separately; see c subdirectory Makefile for details.  


## Usage
See each resulting command-line tool for help (use -h or --help option).  
For example:  
```
/opt/codee/speech/bin/sad_thresh --help  
```


## List of functions
All: F0 SAD VAD
F0: f0_ccs  
SAD: sad_thresh  
VAD: vad_ccs  


## Contributing
This is currently only to view the project in progress.


## License
[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)
