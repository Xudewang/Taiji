Server-side cross match
======================================================================

Positionally cross-match a catalog to the HSC database.

This script reads a catalog to generate an SQL query, which will be output to stdout.
The output query can be posted to the DB server by means of hscReleaseQuery.py
(../hscReleaseQuery/hscReleaseQuery.py).
Thus:

```
hscSspCrossMatch.py catalog.fits | ../hscReleaseQuery/hscReleaseQuery.py --user=USER@local -
```
