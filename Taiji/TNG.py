import glob
import os
import string
import sys
import time

import h5py
import illustris_python as il
import numpy as np
import pandas as pd


def Subhalos_Catalogue(
    basePath, snap, mstar_lower=0, mstar_upper=np.inf, little_h=0.6774
):
    """
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
    This function is from Connor Bottrell's IllustrisTNG repository.
    """
    # convert to units used by TNG (1e10 Msun/h)
    mstar_lower = 10 ** (mstar_lower) / 1e10 * little_h
    mstar_upper = 10 ** (mstar_upper) / 1e10 * little_h

    fields = [
        "SubhaloMassType",
        "SubhaloMassInRadType",
        "SubhaloMassInHalfRadType",
        "SubhaloFlag",
        "SubhaloSFR",
        "SubhaloSFRinHalfRad",
        "SubhaloSFRinRad",
        "SubhaloHalfmassRadType",
    ]

    ptNumStars = il.snapshot.partTypeNum("stars")
    ptNumGas = il.snapshot.partTypeNum("gas")
    ptNumDM = il.snapshot.partTypeNum("dm")

    cat = il.groupcat.loadSubhalos(basePath, snap, fields=fields)
    hdr = il.groupcat.loadHeader(basePath, snap)
    redshift, a2 = hdr["Redshift"], hdr["Time"]
    subs = np.arange(cat["count"], dtype=int)
    flags = cat["SubhaloFlag"]
    mstar = cat["SubhaloMassType"][:, ptNumStars]
    cat["SubfindID"] = subs
    cat["SnapNum"] = [snap for sub in subs]
    subs = subs[(flags != 0) * (mstar >= mstar_lower) * (mstar <= mstar_upper)]

    cat["SubhaloHalfmassRadType_stars"] = (
        cat["SubhaloHalfmassRadType"][:, ptNumStars] * a2 / little_h
    )
    cat["SubhaloHalfmassRadType_gas"] = (
        cat["SubhaloHalfmassRadType"][:, ptNumGas] * a2 / little_h
    )
    cat["SubhaloHalfmassRadType_dm"] = (
        cat["SubhaloHalfmassRadType"][:, ptNumDM] * a2 / little_h
    )
    cat["SubhaloMassInRadType_stars"] = np.log10(
        cat["SubhaloMassInRadType"][:, ptNumStars] * 1e10 / little_h
    )
    cat["SubhaloMassInRadType_gas"] = np.log10(
        cat["SubhaloMassInRadType"][:, ptNumGas] * 1e10 / little_h
    )
    cat["SubhaloMassInRadType_dm"] = np.log10(
        cat["SubhaloMassInRadType"][:, ptNumDM] * 1e10 / little_h
    )
    cat["SubhaloMassInHalfRadType_stars"] = np.log10(
        cat["SubhaloMassInHalfRadType"][:, ptNumStars] * 1e10 / little_h
    )
    cat["SubhaloMassInHalfRadType_gas"] = np.log10(
        cat["SubhaloMassInHalfRadType"][:, ptNumGas] * 1e10 / little_h
    )
    cat["SubhaloMassInHalfRadType_dm"] = np.log10(
        cat["SubhaloMassInHalfRadType"][:, ptNumDM] * 1e10 / little_h
    )
    cat["SubhaloMassType_stars"] = np.log10(
        cat["SubhaloMassType"][:, ptNumStars] * 1e10 / little_h
    )
    cat["SubhaloMassType_gas"] = np.log10(
        cat["SubhaloMassType"][:, ptNumGas] * 1e10 / little_h
    )
    cat["SubhaloMassType_dm"] = np.log10(
        cat["SubhaloMassType"][:, ptNumDM] * 1e10 / little_h
    )
    del (
        cat["SubhaloMassType"],
        cat["SubhaloHalfmassRadType"],
        cat["SubhaloMassInRadType"],
        cat["SubhaloMassInHalfRadType"],
        cat["count"],
    )
    cat = pd.DataFrame.from_dict(cat)
    cat = cat.loc[subs]
    return cat


def Subhalos_SQL(
    cat,
    sim,
    snap,
    host="nantai.ipmu.jp",
    user="bottrell",
    cnf_path="/home/connor.bottrell/.mysql/nantai.cnf",
):
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

    database = f"Illustris{sim}".replace("-", "_")
    columns = list(cat.columns)
    columns.remove("SubhaloFlag")
    db = pymysql.connect(
        host=host,
        user=user,
        database=database,
        read_default_file=cnf_path,
        autocommit=True,
    )
    c = db.cursor()
    for idx in cat.index:
        rec = cat.loc[idx]
        values = [str(rec[col]) for col in columns]
        values = ",".join(values)
        values = values.replace("nan", "-99")
        values = values.replace("-inf", "-99")
        values = values.replace("inf", "-99")
        dbID = f'{rec["SnapNum"]}_{rec["SubfindID"]}'
        dbcmd = [
            f"INSERT INTO Subhalos",
            "(dbID,",
            ",".join(columns),
            ")",
            f"VALUES",
            f'("{dbID}",',
            values,
            ")",
        ]
        dbcmd = " ".join(dbcmd)
        try:
            c.execute(dbcmd)
        except:
            continue
    c.close()
    db.close()
    return


def Rotate(x, axis):
    """return x-vector in instrument frame"""
    Xinst, Yinst, Zinst = _get_inst(axis)
    rx = np.zeros_like(x)
    rx[:, 0] = np.dot(x, Xinst)
    rx[:, 1] = np.dot(x, Yinst)
    rx[:, 2] = np.dot(x, Zinst)
    return rx


def _get_angles(axis):
    """return instrument coordinates in spherical coordinates"""
    angles = {
        "v0": (109.5, 0.0, 0.0),
        "v1": (109.5, 120.0, 0.0),
        "v2": (109.5, -120.0, 0.0),
        "v3": (0.0, 0.0, 0.0),
    }
    if len(axis) == 2:
        inclination, azimuth, posang = angles[axis]
    elif len(axis) == 5:
        inclination = float(axis[1:3])
        azimuth = 0.0
        posang = float(axis[3:])
    return inclination, azimuth, posang


def _get_inst(axis):
    """return instrument coordinates in box coordinates"""
    inclination, azimuth, posang = _get_angles(axis)
    costheta = np.cos(np.radians(inclination))
    sintheta = np.sin(np.radians(inclination))
    cosphi = np.cos(np.radians(azimuth))
    sinphi = np.sin(np.radians(azimuth))
    cospa = np.cos(np.radians(posang))
    sinpa = np.sin(np.radians(posang))

    Xinst = np.array(
        [
            -cosphi * costheta * cospa - sinphi * sinpa,
            -sinphi * costheta * cospa + cosphi * sinpa,
            sintheta * cospa,
        ]
    )
    Yinst = np.array(
        [
            cosphi * costheta * sinpa - sinphi * cospa,
            sinphi * costheta * sinpa + cosphi * cospa,
            -sintheta * sinpa,
        ]
    )
    Zinst = np.array([sintheta * cosphi, sintheta * sinphi, costheta])

    Xinst /= np.sqrt(np.sum(Xinst**2))
    Yinst /= np.sqrt(np.sum(Yinst**2))
    Zinst /= np.sqrt(np.sum(Zinst**2))

    return Xinst, Yinst, Zinst


def included_angle_and_cosine(v1, v2):
    """return the included angle (in radians) and cosine between two vectors"""
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # cosine = np.abs(cosine)
    angle = np.arccos(cosine) * 180 / np.pi

    if angle > 90:
        angle = 180 - angle
        cosine = -cosine

    return angle, cosine
