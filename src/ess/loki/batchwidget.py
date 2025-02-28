import os
import glob
import pandas as pd
import scipp as sc
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipydatagrid import DataGrid
from IPython.display import display
from ipyfilechooser import FileChooser
from ess import sans
from ess import loki
from ess.sans.types import *

def reduce_loki_batch_preliminary(
    sample_run_file: str,
    transmission_run_file: str,
    background_run_file: str,
    empty_beam_file: str,
    direct_beam_file: str,
    mask_files: list = None,
    correct_for_gravity: bool = True,
    uncertainty_mode = UncertaintyBroadcastMode.upper_bound,
    return_events: bool = False,
    wavelength_min: float = 1.0,
    wavelength_max: float = 13.0,
    wavelength_n: int = 201,
    q_start: float = 0.01,
    q_stop: float = 0.3,
    q_n: int = 101
):
    if mask_files is None:
        mask_files = []
    # Define wavelength and Q bins.
    wavelength_bins = sc.linspace("wavelength", wavelength_min, wavelength_max, wavelength_n, unit="angstrom")
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
        raise FileNotFoundError(f"Could not find direct-beam file matching pattern {pattern}")


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
            description="Ebeam SANS:"
        )
        self.ebeam_trans_widget = widgets.Text(
            value="",
            placeholder="Enter Ebeam TRANS run number",
            description="Ebeam TRANS:"
        )
        self.load_csv_button = widgets.Button(description="Load CSV")
        self.load_csv_button.on_click(self.load_csv)
        self.table = DataGrid(pd.DataFrame([]), editable=True, auto_fit_columns=True)
        self.reduce_button = widgets.Button(description="Reduce")
        self.reduce_button.on_click(self.run_reduction)
        # New Clear Log button.
        self.clear_log_button = widgets.Button(description="Clear Log")
        self.clear_log_button.on_click(self.clear_log)
        self.log_output = widgets.Output()
        self.main = widgets.VBox([
            widgets.HBox([self.csv_chooser, self.input_dir_chooser, self.output_dir_chooser]),
            widgets.HBox([self.ebeam_sans_widget, self.ebeam_trans_widget]),
            self.load_csv_button,
            self.table,
            widgets.HBox([self.reduce_button, self.clear_log_button]),
            self.log_output
        ])
    
    def clear_log(self, _):
        self.log_output.clear_output()
    
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
            background_run_file = find_file(input_dir, self.ebeam_sans_widget.value, extension=".nxs")
            empty_beam_file = find_file(input_dir, self.ebeam_trans_widget.value, extension=".nxs")
            with self.log_output:
                print("Using empty-beam files:")
                print("  Background (Ebeam SANS):", background_run_file)
                print("  Empty-beam (Ebeam TRANS):", empty_beam_file)
        except Exception as e:
            with self.log_output:
                print("Error finding empty-beam files:", e)
            return
        df = self.table.data
        for idx, row in df.iterrows():
            sample = row["SAMPLE"]
            try:
                sample_run_file = find_file(input_dir, str(row["SANS"]), extension=".nxs")
                transmission_run_file = find_file(input_dir, str(row["TRANS"]), extension=".nxs")
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
                    mask_files=[mask_file]
                )
            except Exception as e:
                with self.log_output:
                    print(f"Reduction failed for sample {sample}: {e}")
                continue
            out_xye = os.path.join(output_dir, os.path.basename(sample_run_file).replace(".nxs", ".xye"))
            try:
                save_xye_pandas(res["IofQ"], out_xye)
                with self.log_output:
                    print(f"Saved reduced data to {out_xye}")
            except Exception as e:
                with self.log_output:
                    print(f"Failed to save reduced data for {sample}: {e}")
            wavelength_bins = sc.linspace("wavelength", 1.0, 13.0, 201, unit="angstrom")
            x_wl = 0.5 * (wavelength_bins.values[:-1] + wavelength_bins.values[1:])
            plt.figure()
            plt.plot(x_wl, res["transmission"].values, marker='o', linestyle='-')
            plt.title(f"Transmission: {os.path.basename(sample_run_file)}")
            plt.xlabel("Wavelength (angstrom)")
            plt.ylabel("Transmission")
            trans_png = os.path.join(output_dir, os.path.basename(sample_run_file).replace(".nxs", "_transmission.png"))
            plt.savefig(trans_png)
            plt.close()
            q_bins = sc.linspace("Q", 0.01, 0.3, 101, unit="1/angstrom")
            x_q = 0.5 * (q_bins.values[:-1] + q_bins.values[1:])
            plt.figure()
            if res["IofQ"].variances is not None:
                yerr = np.sqrt(res["IofQ"].variances)
                plt.errorbar(x_q, res["IofQ"].values, yerr=yerr, marker='o', linestyle='-')
            else:
                plt.plot(x_q, res["IofQ"].values, marker='o', linestyle='-')
            plt.title(f"I(Q): {os.path.basename(sample_run_file)}")
            plt.xlabel("Q (1/angstrom)")
            plt.ylabel("I(Q)")
            plt.xscale("log")
            plt.yscale("log")
            iq_png = os.path.join(output_dir, os.path.basename(sample_run_file).replace(".nxs", "_IofQ.png"))
            plt.savefig(iq_png)
            plt.close()
            with self.log_output:
                print(f"Reduced sample {sample} and saved outputs.")
    
    @property
    def widget(self):
        return self.main
