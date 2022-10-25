#!/bin/sh
# This script is distributed under Creative Commons 0 (CC0).
# To the extent possible under law, the person who associated CC0
# with this work has waived all copyright and related or neighboring
# rights to this work.

""":"
if command -v python3 &> /dev/null
then
    exec python3 "$0" "$@"
elif command -v python &> /dev/null
then
    exec python "$0" "$@"
else
    echo "Python interpreter not found." 1>&2
fi
exit 1
":"""

from __future__ import print_function, unicode_literals

try:
    import astropy.io.fits as pyfits
except ImportError as e:
    try:
        import pyfits
    except ImportError:
        raise e

import numpy

import argparse
import io
import re
import subprocess
import time
import warnings

warnings.filterwarnings("ignore", module="astropy.io.fits.card")


def startup():
    parser = argparse.ArgumentParser(description="""
        Show a specific mask plane in a FITS file.
    """)
    parser.add_argument("fits", nargs="?", type=fits_load_image, help="""
        Input FITS file.
    """)
    parser.add_argument("masks", metavar="mask[=color]", nargs="*", type=maskcolor_parse, help="""
        Name of mask to show. `color` may be #rgb or #rrggbb or a color name.
    """)
    parser.add_argument("--mask", type=fits_load_mask, help="""
        Mask image alternative to what is contained in the input FITS file.
    """)
    parser.add_argument("--transparency", type=int, default=60, help="""
        Mask transparency (in [0, 100], default 60).
    """)
    parser.add_argument("--no-photoshop", action="store_true", help="""
        Don't do any color corrections to the image.
    """)
    parser.add_argument("--show", metavar="(colors|masks)", choices=["colors", "masks"], help="""
        Show color names or mask names, and exit.
    """)

    args = parser.parse_args()

    if args.show == "colors":
        return show_colors()

    if args.fits is None:
        raise RuntimeError("Specify a FITS file.")

    if args.mask is not None:
        args.fits = (args.fits[0], args.mask)

    if args.fits[1] is None:
        raise RuntimeError("Specify a mask by means of --mask=")

    if args.show == "masks":
        return show_masks(args.fits[1])

    if not args.masks:
        raise RuntimeError("Specify masks to show.")

    return main(
        args.fits[0],
        args.fits[1],
        args.masks,
        transparency = args.transparency,
        photoshop = not args.no_photoshop,
    )


def main(imagehdu, maskhdu, masks, transparency=60, photoshop=True):
    """
    @param imagehdu: a header-data unit of a FITS
    @param maskhdu: a header-data unit of a FITS
    @param masks (List[Tuple[str, ColorSpec]]): list of mask-name and its color
    """
    maskheader = maskhdu.header
    masks = [(maskheader[name], color) for name, color in masks]

    transparency = max(0, min(int(transparency), 100))

    ds9 = Ds9()
    ds9.set_fits(fits_to_bytes(imagehdu))

    for maskbit, color in masks:
        data = ((maskhdu.data & (1 << maskbit)) != 0).astype(numpy.uint8)
        ds9.set_mask(fits_to_bytes(pyfits.PrimaryHDU(data=data)), str(color))

    ds9.xpaset(["mask", "transparency", str(transparency)])

    if photoshop:
        ds9.xpaset(["zscale"])


def show_colors():
    for name, value in sorted(ColorSpec.csscolors.items()):
        print(name, ":", value)


def show_masks(hdu):
    masks = sorted(key for key in hdu.header.keys() if key.startswith("MP_"))
    for mask in masks:
        print(mask)


def fits_load_image(path):
    """
    @param path (str): Path to a FITS file.
    @return The image HDU and the mask HDU in the FITS file.
    """
    m = re.match(r"^(.*)\[([^/\]\\]+)\]$", path)
    if m:
        path, index = m.groups()
        try:
            index = int(index)
        except ValueError:
            pass
    else:
        index = None

    hdus = pyfits.open(path, uint=True)
    if len(hdus) == 1 and (index is None):
        return (hdus[0], None)

    if index is None:
        index = "IMAGE"

    image = None
    mask = None

    for i, hdu in enumerate(hdus):
        header = hdu.header
        exttype = header.get("EXTTYPE") or header.get("EXTNAME")
        if i == index or exttype == index:
            image = hdu
        elif exttype == "MASK":
            mask = fits_ensure_int_image(hdu)

    if image:
        return (image, mask)

    raise RuntimeError("FITS file doesn't contain an image hdu: " + path)


def fits_load_mask(path):
    """
    @param path (str): Path to a FITS file.
    @return The mask HDU in the FITS file.
    """
    m = re.match(r"^(.*)\[([^/\]\\]+)\]$", path)
    if m:
        path, index = m.groups()
        try:
            index = int(index)
        except ValueError:
            pass
    else:
        index = None

    hdus = pyfits.open(path, uint=True)
    if len(hdus) == 1 and (index is None):
        return hdus[0]

    if index is None:
        index = "MASK"

    mask = None

    for i, hdu in enumerate(hdus):
        header = hdu.header
        exttype = header.get("EXTTYPE") or header.get("EXTNAME")
        if i == index or exttype == index:
            mask = fits_ensure_int_image(hdu)
            break

    if mask:
        return mask

    raise RuntimeError("FITS file doesn't contain a mask hdu: " + path)


def fits_ensure_int_image(hdu):
    """
    Ensure that hdu is an integer image HDU.

    It may be a bug of astropy but an integer image HDU is sometimes
    converted to float one even if its header doesn't contain BZERO or BSCALE.
    """
    if hdu.data.dtype.kind in "iu":
        # Already integer
        return hdu

    if (hdu.data.dtype.kind == "f"
    and numpy.all(numpy.floor(hdu.data) == hdu.data)
    ):
        leftmost_bit = max(bitplace
            for key, bitplace in hdu.header.items()
            if key.startswith("MP_") and isinstance(bitplace, int)
        )
        if leftmost_bit < 8:
            dtype = numpy.uint8
        elif leftmost_bit < 16:
            dtype = numpy.uint16
        elif leftmost_bit < 32:
            dtype = numpy.uint32
        else:
            dtype = numpy.uint64

        return type(hdu)(data=hdu.data.astype(dtype), header=hdu.header)

    raise RuntimeError("Mask data is not integer.")


def fits_to_bytes(hdus):
    memfile = io.BytesIO()
    hdus.writeto(memfile)
    return memfile.getvalue()


class Ds9(object):
    __slots__ = ("__pipe", )

    def __init__(self):
        self.__pipe = None
        self.launch()

    def launch(self):
        n_tries = 3
        waittime = 10

        if self.__check_xpaaccess():
            return

        for i_try in range(n_tries):
            if (self.__pipe is not None) and (self.__pipe.poll() is not None):
                self.__pipe.wait()
                self.__pipe = None

            if self.__pipe is None:
                self.__pipe = subprocess.Popen(["ds9"])

            for _ in range(waittime):
                if self.__check_xpaaccess():
                    return
                time.sleep(1)

        raise RuntimeError("Failed to launch ds9 .")

    def set_fits(self, data):
        """
        @param data (bytes): FITS file
        """
        self.launch()
        try:
            self.xpaset(["fits"], data)
        except Exception:
            self.xpaset(["fits", "new"], data)

    def set_mask(self, data, color):
        """
        @param data (bytes): FITS file
        @param color ("#rrggbb")
        """
        self.launch()
        self.xpaset(["mask", "color", color])
        self.xpaset(["fits", "mask"], data)

    @staticmethod
    def __check_xpaaccess():
        p = subprocess.Popen(["xpaaccess", "ds9"], stdout=subprocess.PIPE)
        try:
            msg = p.stdout.read().decode("utf-8")
        finally:
            p.wait()

        return msg.strip() == "yes"

    @staticmethod
    def xpaset(command, data=b""):
        p = subprocess.Popen(
            ["xpaset", "ds9"] + command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = [x.decode("utf-8") for x in p.communicate(data)]
        finally:
            p.wait()

        if err.startswith("XPA$ERROR"):
            raise RuntimeError(err)


def maskcolor_parse(s):
    """
    @param s (str): "maskname=color"
    @return (Tuple[str, ColorSpec])
    """
    s = s.split("=", 1)
    if len(s) == 1:
        return maskname_normalize(s[0]), ColorSpec("tomato")
    if len(s) == 2:
        return maskname_normalize(s[0]), ColorSpec(s[1])


def maskname_normalize(s):
    """
    @param s (str): Mask name with or without the prefix "MP_". Case insensitive.
    @return (str): "MP_..."
    """
    s = s.upper()
    if s.startswith("MP_"):
        return s
    else:
        return "MP_" + s


class ColorSpec(object):
    __slots__ = ("r", "g", "b") # [0, 255]

    def __init__(self, spec):
        """
        @param spec (str):
              * "#rgb", "#rgb,a"
              * "#rrggbb", "#rrggbb"
              * "name", "name"
        """
        spec = ColorSpec.csscolors.get(spec, spec)
        m = re.match("#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$", spec)
        if m:
            r, g, b = m.groups()
            self.r = int(r, 16)
            self.g = int(g, 16)
            self.b = int(b, 16)
            return
        m = re.match("#([0-9a-f])([0-9a-f])([0-9a-f])$", color)
        if m:
            r, g, b = m.groups()
            self.r = int(r + r, 16)
            self.g = int(g + g, 16)
            self.b = int(b + b, 16)
            return

    def __str__(self):
        return "#{self.r:02x}{self.g:02x}{self.b:02x}".format(self=self)

    csscolors = {
        "black": "#000000",
        "silver": "#c0c0c0",
        "gray": "#808080",
        "white": "#ffffff",
        "maroon": "#800000",
        "red": "#ff0000",
        "purple": "#800080",
        "fuchsia": "#ff00ff",
        "green": "#008000",
        "lime": "#00ff00",
        "olive": "#808000",
        "yellow": "#ffff00",
        "navy": "#000080",
        "blue": "#0000ff",
        "teal": "#008080",
        "aqua": "#00ffff",
        "orange": "#ffa500",
        "aliceblue": "#f0f8ff",
        "antiquewhite": "#faebd7",
        "aquamarine": "#7fffd4",
        "azure": "#f0ffff",
        "beige": "#f5f5dc",
        "bisque": "#ffe4c4",
        "blanchedalmond": "#ffebcd",
        "blueviolet": "#8a2be2",
        "brown": "#a52a2a",
        "burlywood": "#deb887",
        "cadetblue": "#5f9ea0",
        "chartreuse": "#7fff00",
        "chocolate": "#d2691e",
        "coral": "#ff7f50",
        "cornflowerblue": "#6495ed",
        "cornsilk": "#fff8dc",
        "crimson": "#dc143c",
        "cyan": "#00ffff",
        "darkblue": "#00008b",
        "darkcyan": "#008b8b",
        "darkgoldenrod": "#b8860b",
        "darkgray": "#a9a9a9",
        "darkgreen": "#006400",
        "darkgrey": "#a9a9a9",
        "darkkhaki": "#bdb76b",
        "darkmagenta": "#8b008b",
        "darkolivegreen": "#556b2f",
        "darkorange": "#ff8c00",
        "darkorchid": "#9932cc",
        "darkred": "#8b0000",
        "darksalmon": "#e9967a",
        "darkseagreen": "#8fbc8f",
        "darkslateblue": "#483d8b",
        "darkslategray": "#2f4f4f",
        "darkslategrey": "#2f4f4f",
        "darkturquoise": "#00ced1",
        "darkviolet": "#9400d3",
        "deeppink": "#ff1493",
        "deepskyblue": "#00bfff",
        "dimgray": "#696969",
        "dimgrey": "#696969",
        "dodgerblue": "#1e90ff",
        "firebrick": "#b22222",
        "floralwhite": "#fffaf0",
        "forestgreen": "#228b22",
        "gainsboro": "#dcdcdc",
        "ghostwhite": "#f8f8ff",
        "gold": "#ffd700",
        "goldenrod": "#daa520",
        "greenyellow": "#adff2f",
        "grey": "#808080",
        "honeydew": "#f0fff0",
        "hotpink": "#ff69b4",
        "indianred": "#cd5c5c",
        "indigo": "#4b0082",
        "ivory": "#fffff0",
        "khaki": "#f0e68c",
        "lavender": "#e6e6fa",
        "lavenderblush": "#fff0f5",
        "lawngreen": "#7cfc00",
        "lemonchiffon": "#fffacd",
        "lightblue": "#add8e6",
        "lightcoral": "#f08080",
        "lightcyan": "#e0ffff",
        "lightgoldenrodyellow": "#fafad2",
        "lightgray": "#d3d3d3",
        "lightgreen": "#90ee90",
        "lightgrey": "#d3d3d3",
        "lightpink": "#ffb6c1",
        "lightsalmon": "#ffa07a",
        "lightseagreen": "#20b2aa",
        "lightskyblue": "#87cefa",
        "lightslategray": "#778899",
        "lightslategrey": "#778899",
        "lightsteelblue": "#b0c4de",
        "lightyellow": "#ffffe0",
        "limegreen": "#32cd32",
        "linen": "#faf0e6",
        "magenta": "#ff00ff",
        "mediumaquamarine": "#66cdaa",
        "mediumblue": "#0000cd",
        "mediumorchid": "#ba55d3",
        "mediumpurple": "#9370db",
        "mediumseagreen": "#3cb371",
        "mediumslateblue": "#7b68ee",
        "mediumspringgreen": "#00fa9a",
        "mediumturquoise": "#48d1cc",
        "mediumvioletred": "#c71585",
        "midnightblue": "#191970",
        "mintcream": "#f5fffa",
        "mistyrose": "#ffe4e1",
        "moccasin": "#ffe4b5",
        "navajowhite": "#ffdead",
        "oldlace": "#fdf5e6",
        "olivedrab": "#6b8e23",
        "orangered": "#ff4500",
        "orchid": "#da70d6",
        "palegoldenrod": "#eee8aa",
        "palegreen": "#98fb98",
        "paleturquoise": "#afeeee",
        "palevioletred": "#db7093",
        "papayawhip": "#ffefd5",
        "peachpuff": "#ffdab9",
        "peru": "#cd853f",
        "pink": "#ffc0cb",
        "plum": "#dda0dd",
        "powderblue": "#b0e0e6",
        "rosybrown": "#bc8f8f",
        "royalblue": "#4169e1",
        "saddlebrown": "#8b4513",
        "salmon": "#fa8072",
        "sandybrown": "#f4a460",
        "seagreen": "#2e8b57",
        "seashell": "#fff5ee",
        "sienna": "#a0522d",
        "skyblue": "#87ceeb",
        "slateblue": "#6a5acd",
        "slategray": "#708090",
        "slategrey": "#708090",
        "snow": "#fffafa",
        "springgreen": "#00ff7f",
        "steelblue": "#4682b4",
        "tan": "#d2b48c",
        "thistle": "#d8bfd8",
        "tomato": "#ff6347",
        "turquoise": "#40e0d0",
        "violet": "#ee82ee",
        "wheat": "#f5deb3",
        "whitesmoke": "#f5f5f5",
        "yellowgreen": "#9acd32",
        "rebeccapurple": "#663399",
    }


if __name__ == "__main__":
    startup()
