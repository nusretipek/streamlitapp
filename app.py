import base64
import io
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
else:
    UploadedFile = Any

APP_TITLE = "Dual Elo Rating"
PAGE_LAYOUT = "wide"
STYLE_BLOCK = """
    <style>
    a.save-results-button {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        color: #ffffff !important;
        background-color: #1f77b4;
        text-decoration: none;
        font-size: 0.85rem;
        border: none;
    }
    a.save-results-button:hover {
        background-color: #145a86;
        color: #ffffff !important;
    }
    div[data-testid="stCodeBlock"] {
        height: 530px !important;
        max-height: 530px !important;
        overflow-y: auto !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stCodeBlock"] pre {
        height: 100% !important;
        max-height: 100% !important;
        overflow-y: auto !important;
        margin: 0 !important;
    }
    </style>
"""
DEFAULT_DATASET = "demo.csv"
OPTIMIZATION_OPTIONS = [0, 1, 2]
VERBOSITY_OPTIONS = [0, 1, 2]
OPTIMIZATION_LABELS: Dict[int, str] = {
    0: "Optimized Elo",
    1: "Optimized Elo + k(c)",
    2: "Joint Optimization",
}
VERBOSITY_LABELS: Dict[int, str] = {
    0: "Optimization Level",
    1: "Iteration Level",
    2: "Debug",
}
DEFAULT_RANDOM_PERMUTATIONS = 30
MIN_RANDOM_PERMUTATIONS = 1
DEFAULT_TOP_N = 10
MIN_TOP_N = 5
MAX_TOP_N = 20
TABLE_HEIGHT = 530


def validate_csv(fileName: str, fileBytes: bytes) -> Optional[pd.DataFrame]:
    """Return a parsed DataFrame when the CSV is valid, otherwise None."""
    if not fileName:
        st.sidebar.error("Uploaded file must have a name.")
        return None

    if not fileName.lower().endswith(".csv"):
        st.sidebar.error("Invalid file type. Please upload a CSV file.")
        return None

    try:
        dataframe = pd.read_csv(io.BytesIO(fileBytes))
    except Exception as exc:
        st.sidebar.error(f"Unable to read CSV file: {exc}")
        return None

    if dataframe.shape[1] < 2:
        st.sidebar.error("CSV file must contain at least two columns for winner and loser.")
        return None

    return dataframe


def configure_page() -> None:
    """Configure the Streamlit page."""
    st.set_page_config(page_title=APP_TITLE, layout=PAGE_LAYOUT)
    st.title(APP_TITLE)
    st.write("Adjust the parameters on the left panel and press 'Run' to execute the program.")


def inject_styles() -> None:
    """Inject CSS customisations into the page."""
    st.markdown(STYLE_BLOCK, unsafe_allow_html=True)


def render_save_link(statusPlaceholder: DeltaGenerator, fileName: str, content: Optional[str]) -> None:
    """Render a download link for the results."""
    if not fileName or content is None:
        statusPlaceholder.empty()
        return

    encodedContent = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    statusPlaceholder.markdown(
        (
            f'<a download="{fileName}" href="data:text/plain;base64,{encodedContent}" '
            f'class="save-results-button">Save results</a>'
        ),
        unsafe_allow_html=True,
    )


def render_results(resultsPlaceholder: DeltaGenerator, resultText: str) -> None:
    if not resultText:
        resultsPlaceholder.empty()
        return

    resultsPlaceholder.code(resultText)


def render_sidebar() -> Tuple[
    Optional[UploadedFile],
    DeltaGenerator,
    int,
    int,
    int,
    int,
    bool,
    bool,
]:
    """Render sidebar inputs and return their values."""
    st.sidebar.title("Input Parameters")
    uploadedFile: Optional[UploadedFile] = st.sidebar.file_uploader(
        "Dataset file",
        type="csv",
        key="dataset_file",
        help=(
            "The CSV file should have only two columns, first is the winner and second is the loser. "
            "Column headers should present, first row is skipped."
        ),
    )

    datasetMessagePlaceholder: DeltaGenerator = st.sidebar.empty()

    optimizationLevel: int = st.sidebar.selectbox(
        "Choose optimization level",
        OPTIMIZATION_OPTIONS,
        index=2,
        format_func=lambda option: OPTIMIZATION_LABELS.get(option, str(option)),
        help="Select the optimization strategy: Optimized Elo, Optimized Elo + k(c), or Joint Optimization.",
    )
    randomPermutations: int = st.sidebar.number_input(
        r"Number of random permutations ($N_{p}$)",
        value=DEFAULT_RANDOM_PERMUTATIONS,
        min_value=MIN_RANDOM_PERMUTATIONS,
        help="Set the number of random permutations for T.",
    )
    topN: int = st.sidebar.number_input(
        r"Top N elements ($N_{t}$)",
        value=DEFAULT_TOP_N,
        min_value=MIN_TOP_N,
        max_value=MAX_TOP_N,
        help=r"Set the number of maximum elements in combinations (5 <= $N_{t}$ <= 20).",
    )
    verbosityLevel: int = st.sidebar.selectbox(
        "Verbosity level",
        VERBOSITY_OPTIONS,
        index=1,
        format_func=lambda level: VERBOSITY_LABELS.get(level, str(level)),
        help="Choose output detail: results only, iteration summaries, or full debug logs.",
    )

    actionColumns = st.sidebar.columns(2)
    runClicked: bool = actionColumns[0].button("Run", use_container_width=True)
    resetClicked: bool = actionColumns[1].button("Reset", use_container_width=True)

    return (
        uploadedFile,
        datasetMessagePlaceholder,
        int(optimizationLevel),
        int(randomPermutations),
        int(topN),
        int(verbosityLevel),
        bool(runClicked),
        bool(resetClicked),
    )


def resolve_dataset(
    uploadedFile: Optional[UploadedFile],
    datasetMessagePlaceholder: DeltaGenerator,
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    """Resolve the dataset to use, falling back to the demo file when needed."""
    csvData: pd.DataFrame = pd.DataFrame()
    datasetLabel: str = DEFAULT_DATASET
    filePath: Optional[str] = None
    usingDefault: bool = False

    if uploadedFile is not None:
        uploadedBytes = uploadedFile.getvalue()
        validatedFrame = validate_csv(uploadedFile.name, uploadedBytes)
        if validatedFrame is not None:
            csvData = validatedFrame
            datasetLabel = uploadedFile.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tempFile:
                tempFile.write(uploadedBytes)
                filePath = tempFile.name
        else:
            datasetMessagePlaceholder.warning(f"Using default file: {DEFAULT_DATASET}")
            usingDefault = True
    else:
        datasetMessagePlaceholder.warning(f"Using default file: {DEFAULT_DATASET}")
        usingDefault = True

    if filePath is None:
        defaultPath = os.path.abspath(DEFAULT_DATASET)
        if os.path.exists(defaultPath):
            csvData = pd.read_csv(defaultPath)
            datasetLabel = Path(defaultPath).name
            filePath = defaultPath
        else:
            datasetMessagePlaceholder.error(f"Default dataset {DEFAULT_DATASET} not found.")
    elif not usingDefault:
        datasetMessagePlaceholder.empty()

    return csvData, datasetLabel, filePath


def build_pivot_table(csvData: pd.DataFrame) -> pd.DataFrame:
    """Build the dominance pivot table from the CSV data."""
    if csvData.empty:
        return pd.DataFrame()

    indices: List[Any] = sorted(set(csvData.iloc[:, 0]).union(set(csvData.iloc[:, 1])))
    pivotTable = (
        pd.crosstab(index=csvData.iloc[:, 0], columns=csvData.iloc[:, 1], dropna=True)
        .reindex(indices, fill_value=0, axis=0)
        .reindex(indices, fill_value=0, axis=1)
    )
    return pivotTable


def execute(
    executablePath: str,
    filePath: str,
    optimizationLevel: int,
    randomPermutations: int,
    topN: int,
    verbosityLevel: int,
    resultsPlaceholder: DeltaGenerator,
) -> Tuple[Optional[int], str]:

    command: List[str] = [
        executablePath,
        "-f",
        filePath,
        "--optimization-level",
        str(optimizationLevel),
        "--n-random",
        str(randomPermutations),
        "--top-n",
        str(topN),
        "--verbose",
        str(verbosityLevel),]

    outputLines: List[str] = []

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,)
            
        if process.stdout is not None:
            for line in process.stdout:
                outputLines.append(line)
                render_results(resultsPlaceholder, "".join(outputLines))
            process.stdout.close()
        returnCode = process.wait()
    except FileNotFoundError as exc:
        st.error(f"Executable not found: {exc}")
        return None, ""
    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
        return None, ""

    finalOutput = "".join(outputLines)
    return returnCode, finalOutput


def render_main_layout(
    csvData: pd.DataFrame,
    pivotTable: pd.DataFrame,
    datasetLabel: str,
    filePath: Optional[str],
    optimizationLevel: int,
    randomPermutations: int,
    topN: int,
    verbosityLevel: int,
    runClicked: bool,
    resetClicked: bool,) -> None:
    
    interactionsColumn: DeltaGenerator
    matrixColumn: DeltaGenerator
    resultsColumn: DeltaGenerator
    interactionsColumn, matrixColumn, resultsColumn = st.columns([1, 2, 3])

    with interactionsColumn:
        st.subheader("Interactions")
        if not csvData.empty:
            st.dataframe(csvData, height=TABLE_HEIGHT)
        else:
            st.info("No dataset available.")

    with matrixColumn:
        st.subheader("Dominance Matrix")
        if not pivotTable.empty:
            st.dataframe(pivotTable, height=TABLE_HEIGHT)
        else:
            st.info("Matrix unavailable.")

    with resultsColumn:
        headerColumns = st.columns([4, 1])
        with headerColumns[0]:
            st.subheader("Results")
        statusPlaceholder: DeltaGenerator = headerColumns[1].empty()
        resultsPlaceholder: DeltaGenerator = st.empty()

        if resetClicked:
            for key in ("dataset_file", "run_output", "save_filename"):
                st.session_state.pop(key, None)
            st.rerun()

        if runClicked:
            statusPlaceholder.markdown("**Computing...**")
            render_results(resultsPlaceholder, "")
            st.session_state.pop("run_output", None)
            st.session_state.pop("save_filename", None)

            executablePath = os.path.abspath("DualEloRating-linux")
            try:
                os.chmod(executablePath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except FileNotFoundError:
                st.error(f"Executable not found: {executablePath}")
                statusPlaceholder.empty()
                return

            if not filePath or not os.path.isfile(filePath):
                st.error(f"File {filePath} not found!")
                statusPlaceholder.empty()
                return

            returnCode, finalOutput = execute(
                executablePath,
                filePath,
                optimizationLevel,
                randomPermutations,
                topN,
                verbosityLevel,
                resultsPlaceholder,)

            if returnCode == 0:
                st.session_state["run_output"] = finalOutput
                st.session_state["save_filename"] = f"{Path(datasetLabel).stem}.txt"
                render_results(resultsPlaceholder, finalOutput)
                render_save_link(statusPlaceholder, st.session_state["save_filename"], finalOutput)
            elif returnCode not in (None, 0):
                st.error(f"Process exited with status {returnCode}.")
                statusPlaceholder.empty()
        else:
            cachedOutput = st.session_state.get("run_output", "")
            render_results(resultsPlaceholder, cachedOutput)
            if cachedOutput and st.session_state.get("save_filename"):
                render_save_link(
                    statusPlaceholder,
                    st.session_state["save_filename"],
                    cachedOutput,)


def main() -> None:
    configure_page()
    inject_styles()

    (uploadedFile,
     datasetMessagePlaceholder,
     optimizationLevel,
     randomPermutations,
     topN,
     verbosityLevel,
     runClicked,
     resetClicked,) = render_sidebar()

    csvData, datasetLabel, filePath = resolve_dataset(uploadedFile, datasetMessagePlaceholder)
    pivotTable = build_pivot_table(csvData)
    render_main_layout(
        csvData,
        pivotTable,
        datasetLabel,
        filePath,
        optimizationLevel,
        randomPermutations,
        topN,
        verbosityLevel,
        runClicked,
        resetClicked,)


if __name__ == "__main__":
    main()
