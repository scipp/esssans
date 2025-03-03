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
from scipp.scipy.interpolate import interp1d
import plopp as pp  # used for plotting in direct beam section

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

# ----------------------------
# Direct Beam Functionality
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
    n_wavelength_bands: int = 50
) -> dict:
    """
    Compute the direct beam function for the LoKI detectors using locally stored data.
    """
    workflow = loki.LokiAtLarmorWorkflow()
    workflow = sans.with_pixel_mask_filenames(workflow, masks=[mask])
    workflow[NeXusDetectorName] = 'larmor_detector'
    
    wl_min = sc.scalar(wavelength_min, unit='angstrom')
    wl_max = sc.scalar(wavelength_max, unit='angstrom')
    workflow[WavelengthBins] = sc.linspace('wavelength', wl_min, wl_max, n_wavelength_bins + 1)
    workflow[WavelengthBands] = sc.linspace('wavelength', wl_min, wl_max, n_wavelength_bands + 1)
    workflow[CorrectForGravity] = True
    workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    workflow[ReturnEvents] = False
    workflow[QBins] = sc.linspace(dim='Q', start=0.01, stop=0.3, num=101, unit='1/angstrom')
    
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
            description="Ebeam SANS:"
        )
        self.ebeam_trans_widget = widgets.Text(
            value="",
            placeholder="Enter Ebeam TRANS run number",
            description="Ebeam TRANS:"
        )
        # Add GUI widgets for reduction parameters:
        self.wavelength_min_widget = widgets.FloatText(value=1.0, description="λ min (Å):")
        self.wavelength_max_widget = widgets.FloatText(value=13.0, description="λ max (Å):")
        self.wavelength_n_widget = widgets.IntText(value=201, description="λ n_bins:")
        self.q_start_widget = widgets.FloatText(value=0.01, description="Q start (1/Å):")
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
        self.main = widgets.VBox([
            widgets.HBox([self.csv_chooser, self.input_dir_chooser, self.output_dir_chooser]),
            widgets.HBox([self.ebeam_sans_widget, self.ebeam_trans_widget]),
            # Reduction parameters:
            widgets.HBox([self.wavelength_min_widget, self.wavelength_max_widget, self.wavelength_n_widget]),
            widgets.HBox([self.q_start_widget, self.q_stop_widget, self.q_n_widget]),
            self.load_csv_button,
            self.table,
            widgets.HBox([self.reduce_button, self.clear_log_button, self.clear_plots_button]),
            self.log_output,
            self.plot_output
        ])
    
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
            background_run_file = find_file(input_dir, self.ebeam_sans_widget.value, extension=".nxs")
            empty_beam_file = find_file(input_dir, self.ebeam_trans_widget.value, extension=".nxs")
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
                    q_n=q_n
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
            fig_trans, ax_trans = plt.subplots()
            ax_trans.plot(x_wl, res["transmission"].values, marker='o', linestyle='-')
            ax_trans.set_title(f"Transmission: {sample} {os.path.basename(sample_run_file)}")
            ax_trans.set_xlabel("Wavelength (Å)")
            ax_trans.set_ylabel("Transmission")
            plt.tight_layout()
            with self.plot_output:
                display(fig_trans)
            trans_png = os.path.join(output_dir, os.path.basename(sample_run_file).replace(".nxs", "_transmission.png"))
            fig_trans.savefig(trans_png, dpi=300)
            plt.close(fig_trans)
            q_bins = sc.linspace("Q", 0.01, 0.3, 101, unit="1/angstrom")
            x_q = 0.5 * (q_bins.values[:-1] + q_bins.values[1:])
            fig_iq, ax_iq = plt.subplots()
            if res["IofQ"].variances is not None:
                yerr = np.sqrt(res["IofQ"].variances)
                ax_iq.errorbar(x_q, res["IofQ"].values, yerr=yerr, marker='o', linestyle='-')
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
            iq_png = os.path.join(output_dir, os.path.basename(sample_run_file).replace(".nxs", "_IofQ.png"))
            fig_iq.savefig(iq_png, dpi=300)
            plt.close(fig_iq)
            with self.log_output:
                print(f"Reduced sample {sample} and saved outputs.")
    
    @property
    def widget(self):
        return self.main

# ----------------------------
# Direct Beam Widget
# ----------------------------
class DirectBeamWidget:
    def __init__(self):
        self.mask_text = widgets.Text(
            value="",
            placeholder="Enter mask file path",
            description="Mask:"
        )
        self.sample_sans_text = widgets.Text(
            value="",
            placeholder="Enter sample SANS file path",
            description="Sample SANS:"
        )
        self.background_sans_text = widgets.Text(
            value="",
            placeholder="Enter background SANS file path",
            description="Background SANS:"
        )
        self.sample_trans_text = widgets.Text(
            value="",
            placeholder="Enter sample TRANS file path",
            description="Sample TRANS:"
        )
        self.background_trans_text = widgets.Text(
            value="",
            placeholder="Enter background TRANS file path",
            description="Background TRANS:"
        )
        self.empty_beam_text = widgets.Text(
            value="",
            placeholder="Enter empty beam file path",
            description="Empty Beam:"
        )
        self.local_Iq_theory_text = widgets.Text(
            value="",
            placeholder="Enter I(q) theory file path",
            description="I(q) Theory:"
        )
        # GUI widgets for direct beam parameters:
        self.db_wavelength_min_widget = widgets.FloatText(value=1.0, description="λ min (Å):")
        self.db_wavelength_max_widget = widgets.FloatText(value=13.0, description="λ max (Å):")
        self.db_n_wavelength_bins_widget = widgets.IntText(value=50, description="λ n_bins:")
        self.db_n_wavelength_bands_widget = widgets.IntText(value=50, description="λ n_bands:")
        
        self.compute_button = widgets.Button(description="Compute Direct Beam")
        self.compute_button.on_click(self.compute_direct_beam)
        self.log_output = widgets.Output()
        self.plot_output = widgets.Output()
        self.main = widgets.VBox([
            self.mask_text,
            self.sample_sans_text,
            self.background_sans_text,
            self.sample_trans_text,
            self.background_trans_text,
            self.empty_beam_text,
            self.local_Iq_theory_text,
            widgets.HBox([
                self.db_wavelength_min_widget,
                self.db_wavelength_max_widget,
                self.db_n_wavelength_bins_widget,
                self.db_n_wavelength_bands_widget
            ]),
            self.compute_button,
            self.log_output,
            self.plot_output
        ])
    
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
            print("  λ min:", wl_min, "λ max:", wl_max, "n_bins:", n_bins, "n_bands:", n_bands)
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
                n_wavelength_bands=n_bands
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
# Build Tabbed Widget
# ----------------------------
reduction_widget = SansBatchReductionWidget().widget
direct_beam_widget = DirectBeamWidget().widget
tabs = widgets.Tab(children=[reduction_widget, direct_beam_widget])
tabs.set_title(0, "Reduction")
tabs.set_title(1, "Direct Beam")

# Display the tab widget.
#display(tabs)
