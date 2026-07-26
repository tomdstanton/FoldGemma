import numpy as np
from foldgemma.data.prep import parse_pdb_string

def test_parse_pdb_string():
    mock_pdb = (
        "ATOM      1  N   MET A   1      27.340  24.430   2.314  1.00 90.50           N  \n"
        "ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 90.50           C  \n"
        "ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00 90.50           C  \n"
        "ATOM      4  CB  MET A   1      25.112  24.880   3.749  1.00 90.50           C  \n"
        "ATOM      5  O   MET A   1      27.886  26.463   4.242  1.00 90.50           O  \n"
        "ATOM      6  N   LYS A   2      26.335  27.830   3.279  1.00 85.25           N  \n"
        "ATOM      7  CA  LYS A   2      26.850  29.136   3.882  1.00 85.25           C  \n"
    )

    aa_seq, plddts = parse_pdb_string(mock_pdb)

    assert aa_seq == b"MK"
    assert len(plddts) == 2
    assert np.allclose(plddts, np.array([90.50, 85.25], dtype=np.float32))

def test_parse_pdb_string_empty():
    aa_seq, plddts = parse_pdb_string("")
    assert aa_seq == b""
    assert len(plddts) == 0
