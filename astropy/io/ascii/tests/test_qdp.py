# Licensed under a 3-clause BSD style license - see LICENSE.rst

import io
import os
import pathlib
import warnings

import numpy as np
import pytest

from astropy import units as u
from astropy.io import ascii
from astropy.io.ascii import qdp
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.exceptions import AstropyUserWarning

DATA = """
! Some initial comments
! More comments
READ TERR 1
READ SERR 3
! Table 0 comment
!a a(pos) a(neg) b be c d
53000.5   0.25  -0.5   1  1.5  3.5 2
54000.5   1.25  -1.5   2  2.5  4.5 3
NO NO NO NO NO
! Table 1 comment
!a a(pos) a(neg) b be c d
54000.5   2.25  -2.5   NO  3.5  5.5 5
55000.5   3.25  -3.5   4  4.5  6.5 nan
"""


def test_line_type():
    assert qdp._line_type("READ SERR 3") == "command"
    assert qdp._line_type(" \n    !some gibberish") == "comment"
    assert qdp._line_type("   ") == "comment"
    assert qdp._line_type(" 21345.45") == "data,1"
    assert qdp._line_type(" 21345.45 1.53e-3 1e-3 .04 NO nan") == "data,6"
    assert qdp._line_type(" 21345.45,1.53e-3,1e-3,.04,NO,nan", delimiter=",") == "data,6"
    assert qdp._line_type(" 21345.45 ! a comment to disturb") == "data,1"
    assert qdp._line_type("NO NO NO NO NO") == "new"
    assert qdp._line_type("NO,NO,NO,NO,NO", delimiter=",") == "new"
    with pytest.raises(ValueError):
        qdp._line_type("N O N NOON OON O")
    with pytest.raises(ValueError):
        qdp._line_type(" some non-comment gibberish")


def test_get_type_from_list_of_lines():
    line0 = "! A comment"
    line1 = "543 12 456.0"
    lines = [line0, line1]
    types, ncol = qdp._get_type_from_list_of_lines(lines)
    assert types[0] == "comment"
    assert types[1] == "data,3"
    assert ncol == 3
    lines.append("23")
    with pytest.raises(ValueError):
        qdp._get_type_from_list_of_lines(lines)


def test_interpret_err_lines():
    col_in = ["MJD", "Rate"]
    cols = qdp._interpret_err_lines(None, 2, names=col_in)
    assert cols[0] == "MJD"
    err_specs = {"terr": [1], "serr": [2]}
    ncols = 5
    cols = qdp._interpret_err_lines(err_specs, ncols, names=col_in)
    assert cols[0] == "MJD"
    assert cols[2] == "MJD_nerr"
    assert cols[4] == "Rate_err"
    with pytest.raises(ValueError):
        qdp._interpret_err_lines(err_specs, 6, names=col_in)


def test_understand_err_col():
    colnames = ["a", "a_err", "b", "b_perr", "b_nerr"]
    serr, terr = qdp._understand_err_col(colnames)
    assert np.allclose(serr, [1])
    assert np.allclose(terr, [2])
    with pytest.raises(ValueError):
        qdp._understand_err_col(["a", "a_nerr"])
    with pytest.raises(ValueError):
        qdp._understand_err_col(["a", "a_perr"])


def test_read_table_qdp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        table = qdp._read_table_qdp(DATA, names=["MJD", "Rate"])
    assert len(table) == 2
    assert table["MJD"][0] == 53000.5
    assert table["MJD_perr"][0] == 0.25
    assert table["MJD_nerr"][0] == -0.5
    assert table["Rate"][0] == 1
    assert table["Rate_err"][0] == 1.5
    assert table.meta["initial_comments"] == [
        "Some initial comments",
        "More comments",
    ]
    assert table.meta["comments"] == ["Table 0 comment"]


def test_read_table_qdp_lowercase():
    lowercase_data = """
    ! Some initial comments
    ! More comments
    read terr 1
    read serr 3
    ! Table 0 comment
    !a a(pos) a(neg) b be c d
    53000.5   0.25  -0.5   1  1.5  3.5 2
    54000.5   1.25  -1.5   2  2.5  4.5 3
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        table = qdp._read_table_qdp(lowercase_data, names=["MJD", "Rate"])
    assert len(table) == 2
    assert table["MJD"][0] == 53000.5
    assert table["MJD_perr"][0] == 0.25
    assert table["MJD_nerr"][0] == -0.5
    assert table["Rate"][0] == 1
    assert table["Rate_err"][0] == 1.5
    assert table.meta["initial_comments"] == [
        "Some initial comments",
        "More comments",
    ]
    assert table.meta["comments"] == ["Table 0 comment"]


def test_write_table_qdp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyUserWarning)
        table = qdp._read_table_qdp(DATA, names=["MJD", "Rate"])
    lines = qdp._write_table_qdp(table)
    assert lines[0] == "!Some initial comments"
    assert lines[1] == "!More comments"
    assert lines[2] == "READ TERR 1"
    assert lines[3] == "READ SERR 3"
    assert lines[4] == "!Table 0 comment"
    assert lines[5] == "!MJD MJD_perr MJD_nerr Rate Rate_err c d"
    assert lines[6] == "53000.5 0.25 -0.5 1 1.5 3.5 2"
    assert lines[7] == "54000.5 1.25 -1.5 2 2.5 4.5 3"


def test_roundtrip(tmp_path):
    example_qdp = """
    ! Initial comment line 1
    ! Initial comment line 2
    READ TERR 1
    READ SERR 3
    ! Table 0 comment
    !a a(pos) a(neg) b be c d
    53000.5   0.25  -0.5   1  1.5  3.5 2
    54000.5   1.25  -1.