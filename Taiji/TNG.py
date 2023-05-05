import glob
import os
import string
import sys
import time

import h5py
import illustris_python as il
import numpy as np
import pandas as pd


def Subhalos_Catalogue(basePath,snap,mstar_lower=0,mstar_upper=np.inf,
                       little_h=0.6774):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
    This function is from Connor Bottrell's IllustrisTNG repository.
    '''
    # convert to units used by TNG (1e10 Msun/h)
    mstar_lower = 10**(mstar_lower)/1e10*little_h
    mstar_upper = 10**(mstar_upper)/1e10*little_h
    
    fields = ['SubhaloMassType','SubhaloMassInRadType',
              'SubhaloMassInHalfRadType',
              'SubhaloFlag','SubhaloSFR',
              'SubhaloSFRinHalfRad','SubhaloSFRinRad',
              'SubhaloHalfmassRadType']

    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumGas = il.snapshot.partTypeNum('gas')
    ptNumDM = il.snapshot.partTypeNum('dm')

    cat = il.groupcat.loadSubhalos(basePath,snap,fields=fields)
    hdr = il.groupcat.loadHeader(basePath,snap)
    redshift,a2 = hdr['Redshift'],hdr['Time']
    subs = np.arange(cat['count'],dtype=int)
    flags = cat['SubhaloFlag']
    mstar = cat['SubhaloMassType'][:,ptNumStars]
    cat['SubfindID'] = subs
    cat['SnapNum'] = [snap for sub in subs]
    subs = subs[(flags!=0)*(mstar>=mstar_lower)*(mstar<=mstar_upper)]

    cat['SubhaloHalfmassRadType_stars'] = cat['SubhaloHalfmassRadType'][:,ptNumStars]*a2/little_h
    cat['SubhaloHalfmassRadType_gas'] = cat['SubhaloHalfmassRadType'][:,ptNumGas]*a2/little_h
    cat['SubhaloHalfmassRadType_dm'] = cat['SubhaloHalfmassRadType'][:,ptNumDM]*a2/little_h
    cat['SubhaloMassInRadType_stars'] = np.log10(cat['SubhaloMassInRadType'][:,ptNumStars]*1e10/little_h)
    cat['SubhaloMassInRadType_gas'] = np.log10(cat['SubhaloMassInRadType'][:,ptNumGas]*1e10/little_h)
    cat['SubhaloMassInRadType_dm'] = np.log10(cat['SubhaloMassInRadType'][:,ptNumDM]*1e10/little_h)
    cat['SubhaloMassInHalfRadType_stars'] = np.log10(cat['SubhaloMassInHalfRadType'][:,ptNumStars]*1e10/little_h)
    cat['SubhaloMassInHalfRadType_gas'] = np.log10(cat['SubhaloMassInHalfRadType'][:,ptNumGas]*1e10/little_h)
    cat['SubhaloMassInHalfRadType_dm'] = np.log10(cat['SubhaloMassInHalfRadType'][:,ptNumDM]*1e10/little_h)
    cat['SubhaloMassType_stars'] = np.log10(cat['SubhaloMassType'][:,ptNumStars]*1e10/little_h)
    cat['SubhaloMassType_gas'] = np.log10(cat['SubhaloMassType'][:,ptNumGas]*1e10/little_h)
    cat['SubhaloMassType_dm'] = np.log10(cat['SubhaloMassType'][:,ptNumDM]*1e10/little_h)
    del cat['SubhaloMassType'], cat['SubhaloHalfmassRadType'], cat['SubhaloMassInRadType'], cat['SubhaloMassInHalfRadType'], cat['count']
    cat = pd.DataFrame.from_dict(cat)
    cat = cat.loc[subs]
    return cat

def Subhalos_SQL(cat,sim,snap,
                 host='nantai.ipmu.jp',user='bottrell',
                 cnf_path='/home/connor.bottrell/.mysql/nantai.cnf'):
    """_summary_

    Args:
        cat (_type_): _description_
        sim (_type_): _description_
        snap (_type_): _description_
        host (str, optional): _description_. Defaults to 'nantai.ipmu.jp'.
        user (str, optional): _description_. Defaults to 'bottrell'.
        cnf_path (str, optional): _description_. Defaults to '/home/connor.bottrell/.mysql/nantai.cnf'.
    The function is from Connor Bottrell's IllustrisTNG repository.
    """
    import pymysql
    database = f'Illustris{sim}'.replace('-','_')
    columns = list(cat.columns)
    columns.remove('SubhaloFlag')
    db = pymysql.connect(host=host,
                         user=user,
                         database=database,
                         read_default_file=cnf_path,
                         autocommit=True
                        )
    c = db.cursor()
    for idx in cat.index:
        rec = cat.loc[idx]
        values = [str(rec[col]) for col in columns]
        values = ','.join(values)
        values = values.replace('nan','-99')
        values = values.replace('-inf','-99')
        values = values.replace('inf','-99')
        dbID = f'{rec["SnapNum"]}_{rec["SubfindID"]}'
        dbcmd = [
            f'INSERT INTO Subhalos',
            '(dbID,',','.join(columns),')',
            f'VALUES',
            f'("{dbID}",',values,')',
        ]
        dbcmd = ' '.join(dbcmd)
        try:
            c.execute(dbcmd)
        except:
            continue
    c.close()
    db.close()
    return