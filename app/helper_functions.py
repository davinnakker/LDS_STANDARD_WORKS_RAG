import polars as pl
import streamlit as st
from typing import Literal


def get_csv(name: Literal['All', 'Old Testament', 'New Testament', 'Book of Mormon', 'Doctrine and Covenants', 'Pearl of Great Price']):
    if name == 'All':
        return "data/st.csv", "data/st_e.npy"
    elif name == 'Old Testament':
        return "data/ot.csv", "data/ot_e.npy"
    elif name == 'New Testament':
        return "data/nt.csv", "data/nt_e.npy"
    elif name == 'Book of Mormon':
        return "data/bom.csv", "data/bom_e.npy"
    elif name == 'Doctrine and Covenants':
        return "data/dc.csv", "data/dc_e.npy"
    elif name == 'Pearl of Great Price':
        return "data/pp.csv", "data/pp_e.npy"

    