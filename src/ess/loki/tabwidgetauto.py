import glob
import os
import re
import threading
import time

import h5py
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plopp as pp  # used for plotting in direct beam section
import scipp as sc
from ipydatagrid import DataGrid
from ipyfilechooser import FileChooser
from IPython.display import display
from ipywidgets import IntSlider, Output
from scipp.scipy.interpolate import interp1d

from ess import loki, sans
from ess.sans.types import *


# ----------------------------
# Reduction Functionality
# ----------------------------
def reduce_loki_batch_preliminary(
    sample_run_file: str,
    transmission_run_file: str,
    background_run_file: str,
    empty_beam_file: str,
    direct_beam_file: str,
    mask_files: list = None,
    correct_for_gravity: bool = True,
    uncertainty_mode=UncertaintyBroadcastMode.upper_bound,
    return_events: bool = False,
    wavelength_min: float = 1.0,
    wavelength_max: float = 13.0,
    wavelength_n: int = 201,
    q_start: float = 0.01,
    q_stop: float = 0.3,
    q_n: int = 101,
):
    if mask_files is None:
        mask_files = []
    # Define wavelength and Q bins.
    wavelength_bins = sc.linspace(
        "wavelength", wavelength_min, wavelength_max, wavelength_n, unit="angstrom"
    )
    q_bins = sc.linspace("Q", q_start, q_stop, q_n, unit="1/angstrom")
    # Initialize the workflow.
    workflow = loki.LokiAtLarmorWorkflow()
    if mask_files:
        workflow = sans.with_pixel_mask_filenames(workflow, masks=mask_files)
    workflow[NeXusDetectorName] = "larmor_detector"
    workflow[WavelengthBins] = wavelength_bins
    workflow[QBins] = q_bins
    workflow[CorrectForGravity] = correct_for_gravity
    workflow[UncertaintyBroadcastMode] = uncertainty_mode
    workflow[ReturnEvents] = return_events
    workflow[Filename[BackgroundRun]] = background_run_file
    workflow[Filename[TransmissionRun[BackgroundRun]]] = transmission_run_file
    workflow[Filename[EmptyBeamRun]] = empty_beam_file
    workflow[DirectBeamFilename] = direct_beam_file
    workflow[Filename[SampleRun]] = sample_run_file
    workflow[Filename[TransmissionRun[SampleRun]]] = transmission_run_file
    center = sans.beam_center_from_center_of_mass(workflow)
    workflow[BeamCenter] = center
    tf = workflow.compute(TransmissionFraction[SampleRun])
    da = workflow.compute(BackgroundSubtractedIofQ)
    return {"transmission": tf, "IofQ": da}


def find_file(work_dir, run_number, extension=".nxs"):
    pattern = os.path.join(work_dir, f"*{run_number}*{extension}")
    files = glob.glob(pattern)
    if files:
        return files[0]
    else:
        raise FileNotFoundError(f"Could not find file matching pattern {pattern}")


def find_direct_beam(work_dir):
    pattern = os.path.join(work_dir, "*direct-beam*.h5")
    files = glob.glob(pattern)
    if files:
        return files[0]
    else:
        raise FileNotFoundError(
            f"Could not find direct-beam file matching pattern {pattern}"
        )


def find_mask_file(work_dir):
    pattern = os.path.join(work_dir, "*mask*.xml")
    files = glob.glob(pattern)
    if files:
        return files[0]
    else:
        raise FileNotFoundError(f"Could not find mask file matching pattern {pattern}")


def save_xye_pandas(data_array, filename):
    q_vals = data_array.coords["Q"].values
    i_vals = data_array.values
    if len(q_vals) != len(i_vals):
        q_vals = 0.5 * (q_vals[:-1] + q_vals[1:])
    if data_array.variances is not None:
        err_vals = np.sqrt(data_array.variances)
        if len(err_vals) != len(i_vals):
            err_vals = 0.5 * (err_vals[:-1] + err_vals[1:])
    else:
        err_vals = np.zeros_like(i_vals)
    df = pd.DataFrame({"Q": q_vals, "I(Q)": i_vals, "Error": err_vals})
    df.to_csv(filename, sep=" ", index=False, header=True)


# ----------------------------
# Helper Functions for Semi-Auto Reduction
# ----------------------------
def extract_run_number(filename):
    m = re.search(r'(\d{4,})', filename)
    if m:
        return m.group(1)
    return ""


def parse_nx_details(filepath):
    details = {}
    with h5py.File(filepath, 'r') as f:
        if 'nicos_details' in f['entry']:
            grp = f['entry']['nicos_details']
            if 'runlabel' in grp:
                val = grp['runlabel'][()]
                details['runlabel'] = (
                    val.decode('utf8') if isinstance(val, bytes) else str(val)
                )
            if 'runtype' in grp:
                val = grp['runtype'][()]
                details['runtype'] = (
                    val.decode('utf8') if isinstance(val, bytes) else str(val)
                )
    return details


# ----------------------------
# Semi-Auto Reduction Widget (unchanged)
# ----------------------------
class SemiAutoReductionWidget:
    def __init__(self):
        self.input_dir_chooser = FileChooser(select_dir=True)
        self.input_dir_chooser.title = "Select Input Folder"
        self.output_dir_chooser = FileChooser(select_dir=True)
        self.output_dir_chooser.title = "Select Output Folder"

        self.scan_button = widgets.Button(description="Scan Directory")
        self.scan_button.on_click(self.scan_directory)

        self.table = DataGrid(pd.DataFrame([]), editable=True, auto_fit_columns=True)

        self.add_row_button = widgets.Button(description="Add Row")
        self.add_row_button.on_click(self.add_row)
        self.delete_row_button = widgets.Button(description="Delete Last Row")
        self.delete_row_button.on_click(self.delete_last_row)

        self.lambda_min_widget = widgets.FloatText(value=1.0, description="λ min (Å):")
        self.lambda_max_widget = widgets.FloatText(value=13.0, description="λ max (Å):")
        self.lambda_n_widget = widgets.IntText(value=201, description="λ n_bins:")
        self.q_min_widget = widgets.FloatText(value=0.01, description="Qmin (1/Å):")
        self.q_max_widget = widgets.FloatText(value=0.3, description="Qmax (1/Å):")
        self.q_n_widget = widgets.IntText(value=101, description="Q n_bins:")

        self.empty_beam_sans_text = widgets.Text(
            value="", description="Ebeam SANS:", disabled=True
        )
        self.empty_beam_trans_text = widgets.Text(
            value="", description="Ebeam TRANS:", disabled=True
        )

        self.reduce_button = widgets.Button(description="Reduce")
        self.reduce_button.on_click(self.run_reduction)

        self.clear_log_button = widgets.Button(description="Clear Log")
        self.clear_log_button.on_click(lambda _: self.log_output.clear_output())
        self.clear_plots_button = widgets.Button(description="Clear Plots")
        self.clear_plots_button.on_click(lambda _: self.plot_output.clear_output())

        self.log_output = widgets.Output()
        self.plot_output = widgets.Output()

        self.main = widgets.VBox(
            [
                widgets.HBox([self.input_dir_chooser, self.output_dir_chooser]),
                self.scan_button,
                self.table,
                widgets.HBox([self.add_row_button, self.delete_row_button]),
                widgets.HBox(
                    [
                        self.lambda_min_widget,
                        self.lambda_max_widget,
                        self.lambda_n_widget,
                    ]
                ),
                widgets.HBox([self.q_min_widget, self.q_max_widget, self.q_n_widget]),
                widgets.HBox([self.empty_beam_sans_text, self.empty_beam_trans_text]),
                widgets.HBox(
                    [self.reduce_button, self.clear_log_button, self.clear_plots_button]
                ),
                self.log_output,
                self.plot_output,
            ]
        )

    def add_row(self, _):
        df = self.table.data
        if df.empty:
            new_row = {'SAMPLE': '', 'SANS': '', 'TRANS': ''}
        else:
            new_row = {col: "" for col in df.columns}
        df = df.append(new_row, ignore_index=True)
        self.table.data = df

    def delete_last_row(self, _):
        df = self.table.data
        if not df.empty:
            df = df.iloc[:-1]
            self.table.data = df

    def scan_directory(self, _):
        self.log_output.clear_output()
        input_dir = self.input_dir_chooser.selected
        if not input_dir or not os.path.isdir(input_dir):
            with self.log_output:
                print("Invalid input folder.")
            return
        nxs_files = glob.glob(os.path.join(input_dir, "*.nxs"))
        groups = {}
        for f in nxs_files:
            try:
                details = parse_nx_details(f)
            except Exception:
                continue
            if 'runlabel' not in details or 'runtype' not in details:
                continue
            runlabel = details['runlabel']
            runtype = details['runtype'].lower()
            run_number = extract_run_number(os.path.basename(f))
            if runlabel not in groups:
                groups[runlabel] = {}
            groups[runlabel][runtype] = run_number
        table_rows = []
        for runlabel, d in groups.items():
            if 'sans' in d and 'trans' in d:
                table_rows.append(
                    {'SAMPLE': runlabel, 'SANS': d['sans'], 'TRANS': d['trans']}
                )
        df = pd.DataFrame(table_rows)
        self.table.data = df
        with self.log_output:
            print(f"Scanned {len(nxs_files)} files. Found {len(df)} reduction entries.")
        ebeam_sans_files = []
        ebeam_trans_files = []
        for f in nxs_files:
            try:
                details = parse_nx_details(f)
            except Exception:
                continue
            if 'runtype' in details:
                if details['runtype'].lower() == 'ebeam_sans':
                    ebeam_sans_files.append(f)
                elif details['runtype'].lower() == 'ebeam_trans':
                    ebeam_trans_files.append(f)
        if ebeam_sans_files:
            ebeam_sans_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            self.empty_beam_sans_text.value = ebeam_sans_files[0]
        else:
            self.empty_beam_sans_text.value = ""
        if ebeam_trans_files:
            ebeam_trans_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            self.empty_beam_trans_text.value = ebeam_trans_files[0]
        else:
            self.empty_beam_trans_text.value = ""

    def run_reduction(self, _):
        self.log_output.clear_output()
        self.plot_output.clear_output()
        input_dir = self.input_dir_chooser.selected
        output_dir = self.output_dir_chooser.selected
        if not input_dir or not os.path.isdir(input_dir):
            with self.log_output:
                print("Input folder is not valid.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            with self.log_output:
                print("Output folder is not valid.")
            return
        try:
            direct_beam_file = find_direct_beam(input_dir)
            with self.log_output:
                print("Using direct-beam file:", direct_beam_file)
        except Exception as e:
            with self.log_output:
                print("Direct-beam file not found:", e)
            return
        background_run_file = self.empty_beam_sans_text.value
        empty_beam_file = self.empty_beam_trans_text.value
        if not background_run_file or not empty_beam_file:
            with self.log_output:
                print("Empty beam files not found.")
            return
        lam_min = self.lambda_min_widget.value
        lam_max = self.lambda_max_widget.value
        lam_n = self.lambda_n_widget.value
        q_min = self.q_min_widget.value
        q_max = self.q_max_widget.value
        q_n = self.q_n_widget.value

        df = self.table.data
        for idx, row in df.iterrows():
            sample = row["SAMPLE"]
            sans_run = row["SANS"]
            trans_run = row["TRANS"]
            try:
                sample_run_file = find_file(input_dir, sans_run, extension=".nxs")
                transmission_run_file = find_file(
                    input_dir, trans_run, extension=".nxs"
                )
            except Exception as e:
                with self.log_output:
                    print(f"Skipping sample {sample}: {e}")
                continue
            try:
                mask_file = find_mask_file(input_dir)
                with self.log_output:
                    print(f"Using mask file: {mask_file} for sample {sample}")
            except Exception as e:
                with self.log_output:
                    print(f"Mask file not found for sample {sample}: {e}")
                continue
            with self.log_output:
                print(f"Reducing sample {sample}...")
            try:
                res = reduce_loki_batch_preliminary(
                    sample_run_file=sample_run_file,
                    transmission_run_file=transmission_run_file,
                    background_run_file=background_run_file,
                    empty_beam_file=empty_beam_file,
                    direct_beam_file=direct_beam_file,
                    mask_files=[mask_file],
                    wavelength_min=lam_min,
                    wavelength_max=lam_max,
                    wavelength_n=lam_n,
                    q_start=q_min,
                    q_stop=q_max,
                    q_n=q_n,
                )
            except Exception as e:
                with self.log_output:
                    print(f"Reduction failed for sample {sample}: {e}")
                continue
            out_xye = os.path.join(
                output_dir, os.path.basename(sample_run_file).replace(".nxs", ".xye")
            )
            try:
                save_xye_pandas(res["IofQ"], out_xye)
                with self.log_output:
                    print(f"Saved reduced data to {out_xye}")
            except Exception as e:
                with self.log_output:
                    print(f"Failed to save reduced data for {sample}: {e}")
            # --- Save Transmission Plot ---
            wavelength_bins = sc.linspace(
                "wavelength", lam_min, lam_max, lam_n, unit="angstrom"
            )
            x_wl = 0.5 * (wavelength_bins.values[:-1] + wavelength_bins.values[1:])
            fig_trans, ax_trans = plt.subplots()
            ax_trans.plot(x_wl, res["transmission"].values, marker='o', linestyle='-')
            ax_trans.set_title(
                f"Transmission: {sample} {os.path.basename(sample_run_file)}"
            )
            ax_trans.set_xlabel("Wavelength (Å)")
            ax_trans.set_ylabel("Transmission")
            plt.tight_layout()
            trans_png = os.path.join(
                output_dir,
                os.path.basename(sample_run_file).replace(".nxs", "_transmission.png"),
            )
            fig_trans.savefig(trans_png, dpi=300)
            plt.close(fig_trans)
            # --- Save I(Q) Plot ---
            q_bins = sc.linspace("Q", q_min, q_max, q_n, unit="1/angstrom")
            x_q = 0.5 * (q_bins.values[:-1] + q_bins.values[1:])
            fig_iq, ax_iq = plt.subplots()
            if res["IofQ"].variances is not None:
                yerr = np.sqrt(res["IofQ"].variances)
                ax_iq.errorbar(
                    x_q, res["IofQ"].values, yerr=yerr, marker='o', linestyle='-'
                )
            else:
                ax_iq.plot(x_q, res["IofQ"].values, marker='o', linestyle='-')
            ax_iq.set_title(f"I(Q): {os.path.basename(sample_run_file)} ({sample})")
            ax_iq.set_xlabel("Q (Å$^{-1}$)")
            ax_iq.set_ylabel("I(Q)")
            ax_iq.set_xscale("log")
            ax_iq.set_yscale("log")
            plt.tight_layout()
            iq_png = os.path.join(
                output_dir,
                os.path.basename(sample_run_file).replace(".nxs", "_IofQ.png"),
            )
            fig_iq.savefig(iq_png, dpi=300)
            plt.close(fig_iq)
            with self.log_output:
                print(f"Reduced sample {sample} and saved outputs.")

    @property
    def widget(self):
        return self.main


# ----------------------------
# Direct Beam Functionality and Widget (unchanged)
# ----------------------------
def compute_direct_beam_local(
    mask: str,
    sample_sans: str,
    background_sans: str,
    sample_trans: str,
    background_trans: str,
    empty_beam: str,
    local_Iq_theory: str,
    wavelength_min: float = 1.0,
    wavelength_max: float = 13.0,
    n_wavelength_bins: int = 50,
    n_wavelength_bands: int = 50,
) -> dict:
    workflow = loki.LokiAtLarmorWorkflow()
    workflow = sans.with_pixel_mask_filenames(workflow, masks=[mask])
    workflow[NeXusDetectorName] = 'larmor_detector'

    wl_min = sc.scalar(wavelength_min, unit='angstrom')
    wl_max = sc.scalar(wavelength_max, unit='angstrom')
    workflow[WavelengthBins] = sc.linspace(
        'wavelength', wl_min, wl_max, n_wavelength_bins + 1
    )
    workflow[WavelengthBands] = sc.linspace(
        'wavelength', wl_min, wl_max, n_wavelength_bands + 1
    )
    workflow[CorrectForGravity] = True
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    workflow[ReturnEvents] = False
    workflow[QBins] = sc.linspace(
        dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom'
    )

    workflow[Filename[SampleRun]] = sample_sans
    workflow[Filename[BackgroundRun]] = background_sans
    workflow[Filename[TransmissionRun[SampleRun]]] = sample_trans
    workflow[Filename[TransmissionRun[BackgroundRun]]] = background_trans
    workflow[Filename[EmptyBeamRun]] = empty_beam

    center = sans.beam_center_from_center_of_mass(workflow)
    print("Computed beam center:", center)
    workflow[BeamCenter] = center

    Iq_theory = sc.io.load_hdf5(local_Iq_theory)
    f = interp1d(Iq_theory, 'Q')
    I0 = f(sc.midpoints(workflow.compute(QBins))).data[0]
    print("Computed I0:", I0)

    results = sans.direct_beam(workflow=workflow, I0=I0, niter=6)

    iofq_full = results[-1]['iofq_full']
    iofq_bands = results[-1]['iofq_bands']
    direct_beam_function = results[-1]['direct_beam']

    pp.plot(
        {'reference': Iq_theory, 'data': iofq_full},
        color={'reference': 'darkgrey', 'data': 'C0'},
        norm='log',
    )
    print("Plotted full-range result vs. theoretical reference.")

    return {
        'direct_beam_function': direct_beam_function,
        'iofq_full': iofq_full,
        'Iq_theory': Iq_theory,
    }


class DirectBeamWidget:
    def __init__(self):
        self.mask_text = widgets.Text(
            value="", placeholder="Enter mask file path", description="Mask:"
        )
        self.sample_sans_text = widgets.Text(
            value="",
            placeholder="Enter sample SANS file path",
            description="Sample SANS:",
        )
        self.background_sans_text = widgets.Text(
            value="",
            placeholder="Enter background SANS file path",
            description="Background SANS:",
        )
        self.sample_trans_text = widgets.Text(
            value="",
            placeholder="Enter sample TRANS file path",
            description="Sample TRANS:",
        )
        self.background_trans_text = widgets.Text(
            value="",
            placeholder="Enter background TRANS file path",
            description="Background TRANS:",
        )
        self.empty_beam_text = widgets.Text(
            value="",
            placeholder="Enter empty beam file path",
            description="Empty Beam:",
        )
        self.local_Iq_theory_text = widgets.Text(
            value="",
            placeholder="Enter I(q) Theory file path",
            description="I(q) Theory:",
        )
        self.db_wavelength_min_widget = widgets.FloatText(
            value=1.0, description="λ min (Å):"
        )
        self.db_wavelength_max_widget = widgets.FloatText(
            value=13.0, description="λ max (Å):"
        )
        self.db_n_wavelength_bins_widget = widgets.IntText(
            value=50, description="λ n_bins:"
        )
        self.db_n_wavelength_bands_widget = widgets.IntText(
            value=50, description="λ n_bands:"
        )

        self.compute_button = widgets.Button(description="Compute Direct Beam")
        self.compute_button.on_click(self.compute_direct_beam)
        self.log_output = widgets.Output()
        self.plot_output = widgets.Output()
        self.main = widgets.VBox(
            [
                self.mask_text,
                self.sample_sans_text,
                self.background_sans_text,
                self.sample_trans_text,
                self.background_trans_text,
                self.empty_beam_text,
                self.local_Iq_theory_text,
                widgets.HBox(
                    [
                        self.db_wavelength_min_widget,
                        self.db_wavelength_max_widget,
                        self.db_n_wavelength_bins_widget,
                        self.db_n_wavelength_bands_widget,
                    ]
                ),
                self.compute_button,
                self.log_output,
                self.plot_output,
            ]
        )

    def compute_direct_beam(self, _):
        self.log_output.clear_output()
        self.plot_output.clear_output()
        mask = self.mask_text.value
        sample_sans = self.sample_sans_text.value
        background_sans = self.background_sans_text.value
        sample_trans = self.sample_trans_text.value
        background_trans = self.background_trans_text.value
        empty_beam = self.empty_beam_text.value
        local_Iq_theory = self.local_Iq_theory_text.value
        wl_min = self.db_wavelength_min_widget.value
        wl_max = self.db_wavelength_max_widget.value
        n_bins = self.db_n_wavelength_bins_widget.value
        n_bands = self.db_n_wavelength_bands_widget.value
        with self.log_output:
            print("Computing direct beam with:")
            print("  Mask:", mask)
            print("  Sample SANS:", sample_sans)
            print("  Background SANS:", background_sans)
            print("  Sample TRANS:", sample_trans)
            print("  Background TRANS:", background_trans)
            print("  Empty Beam:", empty_beam)
            print("  I(q) Theory:", local_Iq_theory)
            print(
                "  λ min:",
                wl_min,
                "λ max:",
                wl_max,
                "n_bins:",
                n_bins,
                "n_bands:",
                n_bands,
            )
        try:
            results = compute_direct_beam_local(
                mask,
                sample_sans,
                background_sans,
                sample_trans,
                background_trans,
                empty_beam,
                local_Iq_theory,
                wavelength_min=wl_min,
                wavelength_max=wl_max,
                n_wavelength_bins=n_bins,
                n_wavelength_bands=n_bands,
            )
            with self.log_output:
                print("Direct beam computation complete.")
        except Exception as e:
            with self.log_output:
                print("Error computing direct beam:", e)

    @property
    def widget(self):
        return self.main


# ----------------------------
# New: Auto Reduction Widget (with plot saving)
# ----------------------------
class AutoReductionWidget:
    def __init__(self):
        self.input_dir_chooser = FileChooser(select_dir=True)
        self.input_dir_chooser.title = "Select Input Folder"
        self.output_dir_chooser = FileChooser(select_dir=True)
        self.output_dir_chooser.title = "Select Output Folder"

        self.start_stop_button = widgets.Button(description="Start")
        self.start_stop_button.on_click(self.toggle_running)
        self.status_label = widgets.Label(value="Stopped")

        self.table = DataGrid(pd.DataFrame([]), editable=False, auto_fit_columns=True)
        self.log_output = widgets.Output()

        self.running = False
        self.thread = None
        self.processed = set()  # Track already reduced entries.
        self.empty_beam_sans = None
        self.empty_beam_trans = None

        self.main = widgets.VBox(
            [
                widgets.HBox([self.input_dir_chooser, self.output_dir_chooser]),
                widgets.HBox([self.start_stop_button, self.status_label]),
                self.table,
                self.log_output,
            ]
        )

    def toggle_running(self, _):
        if not self.running:
            self.running = True
            self.start_stop_button.description = "Stop"
            self.status_label.value = "Running"
            self.thread = threading.Thread(target=self.background_loop, daemon=True)
            self.thread.start()
        else:
            self.running = False
            self.start_stop_button.description = "Start"
            self.status_label.value = "Stopped"

    def background_loop(self):
        while self.running:
            input_dir = self.input_dir_chooser.selected
            output_dir = self.output_dir_chooser.selected
            if not input_dir or not os.path.isdir(input_dir):
                with self.log_output:
                    print("Invalid input folder. Waiting for valid selection...")
                time.sleep(10)
                continue
            if not output_dir or not os.path.isdir(output_dir):
                with self.log_output:
                    print("Invalid output folder. Waiting for valid selection...")
                time.sleep(10)
                continue

            # Scan for .nxs files and build the reduction table.
            nxs_files = glob.glob(os.path.join(input_dir, "*.nxs"))
            groups = {}
            for f in nxs_files:
                try:
                    details = parse_nx_details(f)
                except Exception:
                    continue
                if 'runlabel' not in details or 'runtype' not in details:
                    continue
                runlabel = details['runlabel']
                runtype = details['runtype'].lower()
                run_number = extract_run_number(os.path.basename(f))
                if runlabel not in groups:
                    groups[runlabel] = {}
                groups[runlabel][runtype] = run_number
            table_rows = []
            for runlabel, d in groups.items():
                if 'sans' in d and 'trans' in d:
                    table_rows.append(
                        {'SAMPLE': runlabel, 'SANS': d['sans'], 'TRANS': d['trans']}
                    )
            df = pd.DataFrame(table_rows)
            self.table.data = df
            with self.log_output:
                print(
                    f"Scanned {len(nxs_files)} files. Found {len(df)} reduction entries."
                )

            # Identify empty beam files.
            ebeam_sans_files = []
            ebeam_trans_files = []
            for f in nxs_files:
                try:
                    details = parse_nx_details(f)
                except Exception:
                    continue
                if 'runtype' in details:
                    if details['runtype'].lower() == 'ebeam_sans':
                        ebeam_sans_files.append(f)
                    elif details['runtype'].lower() == 'ebeam_trans':
                        ebeam_trans_files.append(f)
            if ebeam_sans_files:
                ebeam_sans_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                self.empty_beam_sans = ebeam_sans_files[0]
            else:
                self.empty_beam_sans = None
            if ebeam_trans_files:
                ebeam_trans_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                self.empty_beam_trans = ebeam_trans_files[0]
            else:
                self.empty_beam_trans = None

            # Get the direct beam file.
            try:
                direct_beam_file = find_direct_beam(input_dir)
            except Exception as e:
                with self.log_output:
                    print("Direct-beam file not found:", e)
                time.sleep(10)
                continue

            # Process new reduction entries.
            for index, row in df.iterrows():
                key = (row["SAMPLE"], row["SANS"], row["TRANS"])
                if key in self.processed:
                    continue
                try:
                    sample_run_file = find_file(
                        input_dir, row["SANS"], extension=".nxs"
                    )
                    transmission_run_file = find_file(
                        input_dir, row["TRANS"], extension=".nxs"
                    )
                except Exception as e:
                    with self.log_output:
                        print(f"Skipping sample {row['SAMPLE']}: {e}")
                    continue
                try:
                    mask_file = find_mask_file(input_dir)
                    with self.log_output:
                        print(
                            f"Using mask file: {mask_file} for sample {row['SAMPLE']}"
                        )
                except Exception as e:
                    with self.log_output:
                        print(f"Mask file not found for sample {row['SAMPLE']}: {e}")
                    continue
                if not self.empty_beam_sans or not self.empty_beam_trans:
                    with self.log_output:
                        print(
                            "Empty beam files not found, skipping reduction for sample",
                            row["SAMPLE"],
                        )
                    continue

                with self.log_output:
                    print(f"Reducing sample {row['SAMPLE']}...")
                try:
                    res = reduce_loki_batch_preliminary(
                        sample_run_file=sample_run_file,
                        transmission_run_file=transmission_run_file,
                        background_run_file=self.empty_beam_sans,
                        empty_beam_file=self.empty_beam_trans,
                        direct_beam_file=direct_beam_file,
                        mask_files=[mask_file],
                        wavelength_min=1.0,
                        wavelength_max=13.0,
                        wavelength_n=201,
                        q_start=0.01,
                        q_stop=0.3,
                        q_n=101,
                    )
                except Exception as e:
                    with self.log_output:
                        print(f"Reduction failed for sample {row['SAMPLE']}: {e}")
                    continue
                out_xye = os.path.join(
                    output_dir,
                    os.path.basename(sample_run_file).replace(".nxs", ".xye"),
                )
                try:
                    save_xye_pandas(res["IofQ"], out_xye)
                    with self.log_output:
                        print(f"Saved reduced data to {out_xye}")
                except Exception as e:
                    with self.log_output:
                        print(f"Failed to save reduced data for {row['SAMPLE']}: {e}")
                # --- Save Transmission Plot ---
                wavelength_bins = sc.linspace(
                    "wavelength", 1.0, 13.0, 201, unit="angstrom"
                )
                x_wl = 0.5 * (wavelength_bins.values[:-1] + wavelength_bins.values[1:])
                fig_trans, ax_trans = plt.subplots()
                ax_trans.plot(
                    x_wl, res["transmission"].values, marker='o', linestyle='-'
                )
                ax_trans.set_title(
                    f"Transmission: {row['SAMPLE']} {os.path.basename(sample_run_file)}"
                )
                ax_trans.set_xlabel("Wavelength (Å)")
                ax_trans.set_ylabel("Transmission")
                plt.tight_layout()
                trans_png = os.path.join(
                    output_dir,
                    os.path.basename(sample_run_file).replace(
                        ".nxs", "_transmission.png"
                    ),
                )
                fig_trans.savefig(trans_png, dpi=300)
                plt.close(fig_trans)
                # --- Save I(Q) Plot ---
                q_bins = sc.linspace("Q", 0.01, 0.3, 101, unit="1/angstrom")
                x_q = 0.5 * (q_bins.values[:-1] + q_bins.values[1:])
                fig_iq, ax_iq = plt.subplots()
                if res["IofQ"].variances is not None:
                    yerr = np.sqrt(res["IofQ"].variances)
                    ax_iq.errorbar(
                        x_q, res["IofQ"].values, yerr=yerr, marker='o', linestyle='-'
                    )
                else:
                    ax_iq.plot(x_q, res["IofQ"].values, marker='o', linestyle='-')
                ax_iq.set_title(
                    f"I(Q): {os.path.basename(sample_run_file)} ({row['SAMPLE']})"
                )
                ax_iq.set_xlabel("Q (Å$^{-1}$)")
                ax_iq.set_ylabel("I(Q)")
                ax_iq.set_xscale("log")
                ax_iq.set_yscale("log")
                plt.tight_layout()
                iq_png = os.path.join(
                    output_dir,
                    os.path.basename(sample_run_file).replace(".nxs", "_IofQ.png"),
                )
                fig_iq.savefig(iq_png, dpi=300)
                plt.close(fig_iq)
                with self.log_output:
                    print(f"Reduced sample {row['SAMPLE']} and saved outputs.")
                self.processed.add(key)
            time.sleep(10)

    @property
    def widget(self):
        return self.main


# ----------------------------
# Widgets for Reduction and Direct Beam
# ----------------------------
class SansBatchReductionWidget:
    def __init__(self):
        self.csv_chooser = FileChooser(select_dir=False)
        self.csv_chooser.title = "Select CSV File"
        self.csv_chooser.filter_pattern = "*.csv"
        self.input_dir_chooser = FileChooser(select_dir=True)
        self.input_dir_chooser.title = "Select Input Folder"
        self.output_dir_chooser = FileChooser(select_dir=True)
        self.output_dir_chooser.title = "Select Output Folder"
        self.ebeam_sans_widget = widgets.Text(
            value="",
            placeholder="Enter Ebeam SANS run number",
            description="Ebeam SANS:",
        )
        self.ebeam_trans_widget = widgets.Text(
            value="",
            placeholder="Enter Ebeam TRANS run number",
            description="Ebeam TRANS:",
        )
        # Add GUI widgets for reduction parameters:
        self.wavelength_min_widget = widgets.FloatText(
            value=1.0, description="λ min (Å):"
        )
        self.wavelength_max_widget = widgets.FloatText(
            value=13.0, description="λ max (Å):"
        )
        self.wavelength_n_widget = widgets.IntText(value=201, description="λ n_bins:")
        self.q_start_widget = widgets.FloatText(
            value=0.01, description="Q start (1/Å):"
        )
        self.q_stop_widget = widgets.FloatText(value=0.3, description="Q stop (1/Å):")
        self.q_n_widget = widgets.IntText(value=101, description="Q n_bins:")

        self.load_csv_button = widgets.Button(description="Load CSV")
        self.load_csv_button.on_click(self.load_csv)
        self.table = DataGrid(pd.DataFrame([]), editable=True, auto_fit_columns=True)
        self.reduce_button = widgets.Button(description="Reduce")
        self.reduce_button.on_click(self.run_reduction)
        self.clear_log_button = widgets.Button(description="Clear Log")
        self.clear_log_button.on_click(self.clear_log)
        self.clear_plots_button = widgets.Button(description="Clear Plots")
        self.clear_plots_button.on_click(self.clear_plots)
        self.log_output = widgets.Output()
        self.plot_output = widgets.Output()
        self.main = widgets.VBox(
            [
                widgets.HBox(
                    [self.csv_chooser, self.input_dir_chooser, self.output_dir_chooser]
                ),
                widgets.HBox([self.ebeam_sans_widget, self.ebeam_trans_widget]),
                # Reduction parameters:
                widgets.HBox(
                    [
                        self.wavelength_min_widget,
                        self.wavelength_max_widget,
                        self.wavelength_n_widget,
                    ]
                ),
                widgets.HBox(
                    [self.q_start_widget, self.q_stop_widget, self.q_n_widget]
                ),
                self.load_csv_button,
                self.table,
                widgets.HBox(
                    [self.reduce_button, self.clear_log_button, self.clear_plots_button]
                ),
                self.log_output,
                self.plot_output,
            ]
        )

    def clear_log(self, _):
        self.log_output.clear_output()

    def clear_plots(self, _):
        self.plot_output.clear_output()

    def load_csv(self, _):
        csv_path = self.csv_chooser.selected
        if not csv_path or not os.path.exists(csv_path):
            with self.log_output:
                print("CSV file not selected or does not exist.")
            return
        df = pd.read_csv(csv_path)
        self.table.data = df
        with self.log_output:
            print(f"Loaded reduction table with {len(df)} rows from {csv_path}.")

    def run_reduction(self, _):
        input_dir = self.input_dir_chooser.selected
        output_dir = self.output_dir_chooser.selected
        if not input_dir or not os.path.isdir(input_dir):
            with self.log_output:
                print("Input folder is not valid.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            with self.log_output:
                print("Output folder is not valid.")
            return
        try:
            direct_beam_file = find_direct_beam(input_dir)
            with self.log_output:
                print("Using direct-beam file:", direct_beam_file)
        except Exception as e:
            with self.log_output:
                print("Direct-beam file not found:", e)
            return
        try:
            background_run_file = find_file(
                input_dir, self.ebeam_sans_widget.value, extension=".nxs"
            )
            empty_beam_file = find_file(
                input_dir, self.ebeam_trans_widget.value, extension=".nxs"
            )
            with self.log_output:
                print("Using empty-beam files:")
                print("  Background (Ebeam SANS):", background_run_file)
                print("  Empty beam (Ebeam TRANS):", empty_beam_file)
        except Exception as e:
            with self.log_output:
                print("Error finding empty beam files:", e)
            return
        # Retrieve reduction parameters from widgets.
        wl_min = self.wavelength_min_widget.value
        wl_max = self.wavelength_max_widget.value
        wl_n = self.wavelength_n_widget.value
        q_start = self.q_start_widget.value
        q_stop = self.q_stop_widget.value
        q_n = self.q_n_widget.value
        df = self.table.data
        for idx, row in df.iterrows():
            sample = row["SAMPLE"]
            try:
                sample_run_file = find_file(
                    input_dir, str(row["SANS"]), extension=".nxs"
                )
                transmission_run_file = find_file(
                    input_dir, str(row["TRANS"]), extension=".nxs"
                )
            except Exception as e:
                with self.log_output:
                    print(f"Skipping sample {sample}: {e}")
                continue
            mask_candidate = str(row.get("mask", "")).strip()
            mask_file = None
            if mask_candidate:
                mask_file_candidate = os.path.join(input_dir, f"{mask_candidate}.xml")
                if os.path.exists(mask_file_candidate):
                    mask_file = mask_file_candidate
            if mask_file is None:
                try:
                    mask_file = find_mask_file(input_dir)
                    with self.log_output:
                        print(f"Identified mask file: {mask_file} for sample {sample}")
                except Exception as e:
                    with self.log_output:
                        print(f"Mask file not found for sample {sample}: {e}")
                    continue
            with self.log_output:
                print(f"Reducing sample {sample}...")
            try:
                res = reduce_loki_batch_preliminary(
                    sample_run_file=sample_run_file,
                    transmission_run_file=transmission_run_file,
                    background_run_file=background_run_file,
                    empty_beam_file=empty_beam_file,
                    direct_beam_file=direct_beam_file,
                    mask_files=[mask_file],
                    wavelength_min=wl_min,
                    wavelength_max=wl_max,
                    wavelength_n=wl_n,
                    q_start=q_start,
                    q_stop=q_stop,
                    q_n=q_n,
                )
            except Exception as e:
                with self.log_output:
                    print(f"Reduction failed for sample {sample}: {e}")
                continue
            out_xye = os.path.join(
                output_dir, os.path.basename(sample_run_file).replace(".nxs", ".xye")
            )
            try:
                save_xye_pandas(res["IofQ"], out_xye)
                with self.log_output:
                    print(f"Saved reduced data to {out_xye}")
            except Exception as e:
                with self.log_output:
                    print(f"Failed to save reduced data for {sample}: {e}")
            wavelength_bins = sc.linspace(
                "wavelength", wl_min, wl_max, wl_n, unit="angstrom"
            )
            x_wl = 0.5 * (wavelength_bins.values[:-1] + wavelength_bins.values[1:])
            fig_trans, ax_trans = plt.subplots()
            ax_trans.plot(x_wl, res["transmission"].values, marker='o', linestyle='-')
            ax_trans.set_title(
                f"Transmission: {sample} {os.path.basename(sample_run_file)}"
            )
            ax_trans.set_xlabel("Wavelength (Å)")
            ax_trans.set_ylabel("Transmission")
            plt.tight_layout()
            with self.plot_output:
                display(fig_trans)
            trans_png = os.path.join(
                output_dir,
                os.path.basename(sample_run_file).replace(".nxs", "_transmission.png"),
            )
            fig_trans.savefig(trans_png, dpi=300)
            plt.close(fig_trans)
            q_bins = sc.linspace("Q", q_start, q_stop, q_n, unit="1/angstrom")
            x_q = 0.5 * (q_bins.values[:-1] + q_bins.values[1:])
            fig_iq, ax_iq = plt.subplots()
            if res["IofQ"].variances is not None:
                yerr = np.sqrt(res["IofQ"].variances)
                ax_iq.errorbar(
                    x_q, res["IofQ"].values, yerr=yerr, marker='o', linestyle='-'
                )
            else:
                ax_iq.plot(x_q, res["IofQ"].values, marker='o', linestyle='-')
            ax_iq.set_title(f"I(Q): {os.path.basename(sample_run_file)} ({sample})")
            ax_iq.set_xlabel("Q (Å$^{-1}$)")
            ax_iq.set_ylabel("I(Q)")
            ax_iq.set_xscale("log")
            ax_iq.set_yscale("log")
            plt.tight_layout()
            with self.plot_output:
                display(fig_iq)
            iq_png = os.path.join(
                output_dir,
                os.path.basename(sample_run_file).replace(".nxs", "_IofQ.png"),
            )
            fig_iq.savefig(iq_png, dpi=300)
            plt.close(fig_iq)
            with self.log_output:
                print(f"Reduced sample {sample} and saved outputs.")

    @property
    def widget(self):
        return self.main


# ----------------------------
# Build the tabbed widget.
# ----------------------------
reduction_widget = SansBatchReductionWidget().widget
direct_beam_widget = DirectBeamWidget().widget
semi_auto_reduction_widget = SemiAutoReductionWidget().widget
auto_reduction_widget = AutoReductionWidget().widget

tabs = widgets.Tab(
    children=[
        direct_beam_widget,
        reduction_widget,
        semi_auto_reduction_widget,
        auto_reduction_widget,
    ]
)
tabs.set_title(0, "Direct Beam")
tabs.set_title(1, "Reduction (Manual)")
tabs.set_title(2, "Reduction (Smart)")
tabs.set_title(3, "Reduction (Auto)")

# display(tabs)
