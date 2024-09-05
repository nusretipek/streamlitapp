import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import tempfile
import stat
import os


# CSV validation
def validate_csv(file):
    if file is not None:
        if file.name.endswith('.csv'):
            return True
        else:
            st.sidebar.error("Invalid file type. Please upload a CSV file.")
            return False
    return True


# Page config
st.set_page_config(page_title="Dual-K Elo Rating", layout="wide")
st.title("Dual-K Elo Rating")
st.write("Adjust the parameters on the left and press 'Run' to execute the program.")

# Sidebar
st.sidebar.title("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Dataset file",
                                         type="csv",
                                         help="The CSV file should have only two columns, "
                                              "first is the winner and second is the loser. "
                                              "With or without headers are accepted.")
if uploaded_file is not None and validate_csv(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
else:
    st.sidebar.warning("Using default file: demo.csv")
    file_path = os.path.abspath("demo.csv")

opt = st.sidebar.selectbox("Choose optimization level", [0, 1, 2],
                           index=0,
                           help="Select the appropriate optimization level (0, 1, or 2)")
initial_k2 = st.sidebar.number_input(r"Initial $k_{c}$",
                                     value=5.298317366548,
                                     min_value=0.00,
                                     max_value=10.00,
                                     help=r"Set the initial $k_{c}$ parameter, in natural logarithm space.",
                                     format="%.4f")
n_random = st.sidebar.number_input(r"Number of random permutations ($N_{p}$)",
                                   value=100,
                                   min_value=1,
                                   help="Set the number of random permutations for T.")
top_n = st.sidebar.number_input(r"Top N elements ($N_{t}$)",
                                value=10,
                                min_value=5,
                                max_value=20,
                                help=r"Set the number of maximum elements in combinations (5 <= $N_{t}$ <= 20).")
verbose = st.sidebar.selectbox("Verbosity level",
                               [0, 1, 2],
                               index=0,
                               help="Set the verbosity level (0, 1, or 2).")

small_col, left_col, right_col = st.columns([1, 2, 3])
with small_col:
    if uploaded_file is not None or os.path.exists(file_path):
        csv_data = pd.read_csv(file_path)
        indices = sorted(set(list(csv_data.iloc[:, 0]) + list(csv_data.iloc[:, 1])))
        pivot = pd.crosstab(index=csv_data.iloc[:, 0],
                            columns=csv_data.iloc[:, 1],
                            dropna=True).reindex(indices,
                                                 fill_value=0,
                                                 axis=0).reindex(indices,
                                                                 fill_value=0,
                                                                 axis=1)
        st.subheader("Interactions")
        st.dataframe(csv_data, height=530)

with left_col:
    st.subheader("Dominance Matrix")
    st.dataframe(pivot, height=530)

with right_col:
    st.subheader("Results")
    if st.sidebar.button("Run"):
        executable_path = os.path.abspath("example")
        os.chmod(executable_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        if not os.path.isfile(file_path):
            st.error(f"File {file_path} not found!")
        else:
            cmd = [executable_path,
                   "-f", file_path,
                   "--optimization-level", str(opt),
                   "--initial-k2", str(initial_k2),
                   "--n-random", str(n_random),
                   "--top-n", str(top_n),
                   "--verbose", str(verbose)]

            with st.spinner("Computing..."):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    stdout = result.stdout
                    stderr = result.stderr
                    st.text_area(label="", label_visibility="collapsed", value=stdout + stderr, height=530)
                except subprocess.CalledProcessError as e:
                    st.error(f"An error occurred: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    if st.sidebar.button("Reset"):
        uploaded_file = None
        st.rerun()
        
# <Checkpoint>
