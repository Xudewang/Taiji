downloadPsf.py
==============================================================================

Download PSF images of coadd images from the website of HSC data release.

Requirements
------------------------------------------------------------------------------

python >= 3.7

Usage
------------------------------------------------------------------------------

### Download images of all bands at a location

```
python3 downloadPsf.py --ra=222.222 --dec=44.444 --name="psf-{tract}-{patch[0]},{patch[1]}-{filter}"
```

Note that `{filter}` must appear in `--name`.
Otherwise, the five images of the five bands will be written
to a single file over and over.

Also, if you do not specify `--tract` and `--patch`,
you should include `{tract}` and `{patch}` in `--name`
so as to know (tract, patch).

### Use coordinate list

You can feed a coordinate list that is in nearly the same format as
https://hsc-release.mtk.nao.ac.jp/psf/pdr3/manual.html#Bulk_mode

There are a few differences:

  - There must not appear comments
    except for the mandatory one at the first line.

  - You can use "all" as a value of "filter" field.

  - The default value of `centered` is True.

It is permissible for the coordinate list to contain only coordinates.
For example:

```
#? ra      dec
222.222  44.444
222.223  44.445
222.224  44.446
```

In this case, you have to specify other fields via the command line:

```
python3 downloadPsf.py \
    --centered=False \
    --name="psf_{tract}_{patch[0]}_{patch[1]}_{ra}_{dec}_{filter}" \
    --list=coordlist.txt # <- the name of the above list
```

It is more efficient to use a list like the example above
than to use a for-loop to call the script iteratively.

### Stop asking a password

To stop the script asking your password, put the password
into an environment variable. (Default: `HSC_SSP_CAS_PASSWORD`)

```
read -s HSC_SSP_CAS_PASSWORD
export HSC_SSP_CAS_PASSWORD
```

Then, run the script with `--user` option:

```
python3 downloadPsf.py \
    --ra=222.222 --dec=44.444 \
    --name="psf-{tract}-{patch[0]},{patch[1]}-{filter}" \
    --user=USERNAME
```

If you are using your own personal laptop or desktop,
you may pass your password through `--password` option.
But you must never do so
if there are other persons using the same computer.
Remember that other persons can see your command lines
with, for example, `top` command.
(If it is GNU's `top`, press `C` key to see others' command lines).

Usage as a python module
------------------------------------------------------------------------------

Here is an example:

```
import downloadPsf

psfreq = downloadPsf.PsfRequest.create(
    ra="11h11m11.111s",
    dec="-1d11m11.111s",
)

images = downloadPsf.download(psfreq)

# Multiple images (of various filters) are returned.
# We look into the first one of them.
metadata, data = images[0]
print(metadata)

# `data` is just the binary data of a FITS file.
# You can use, for example, `astropy` to decode it.
import io
import astropy.io.fits
hdus = astropy.io.fits.open(io.BytesIO(data))
print(hdus)
```
