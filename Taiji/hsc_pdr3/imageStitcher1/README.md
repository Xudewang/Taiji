# Image Stitcher #1
A patch is a 4k x 4k image.  It is relatively large, but one may wish to generate a larger image.  This tool stitches adjacent patches in the same tract together.

## Basic Usage
```sh
python imageStitcher1.py -o stitched.fits calexp-HSC-I-9813-*.fits

ds9 stitched.fits
```

## ```--help```
```
usage: imageStitcher1.py [-h] --out OUT [--binsize BINSIZE] FILE [FILE ...]

This tool stitches adjacent patches in the same tract together.

positional arguments:
  FILE                  patch files to be stitched

optional arguments:
  -h, --help            show this help message and exit
  --out OUT, -o OUT     output file
  --binsize BINSIZE, -b BINSIZE
                        bin size
```

