# Postage Stamps in Color
If you want postage-stamps of your objects in color, you can upload an object list with this tool.

## Basic Usage
```sh
cat > coords.txt <<EOT
# ra         dec             outfile(optional)
33.995270    -5.011639       a.png
33.994442    -4.996707       b.png
33.994669    -5.001553       c.png
33.996395    -5.008107       d.png
33.995679    -4.993945       e.png
33.997352    -5.010902       f.png
33.997315    -5.012523       g.png
33.997438    -5.011647       h.png
33.997379    -5.010878       i.png
33.996636    -5.008742       j.png
EOT

python colorPostage.py --user YOUR_ACCOUNT --outDir pngs coords.txt
```

## Advanced Usage
```
usage: colorPostage.py [-h] --outDir OUTDIR --user USER
                       [--filters FILTERS FILTERS FILTERS] [--fov FOV]
                       [--rerun {any,pdr3_dud,pdr3_wide}] [--color {hsc,sdss}]
                       input

positional arguments:
  input

optional arguments:
  -h, --help            show this help message and exit
  --outDir OUTDIR, -o OUTDIR
  --user USER, -u USER
  --filters FILTERS FILTERS FILTERS, -f FILTERS FILTERS FILTERS
  --fov FOV
  --rerun {any,pdr3_dud,pdr3_wide}
  --color {hsc,sdss}
```