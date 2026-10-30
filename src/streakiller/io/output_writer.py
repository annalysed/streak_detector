"""
OutputWriter protocol and LocalOutputWriter implementation.

The OutputWriter Protocol means any class with a write(result) method can be
used in the pipeline — a future S3OutputWriter just needs to satisfy this
interface without changing any pipeline code.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image as PILImage
import cv2
#from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import map_coordinates

from streakiller.config.schema import FftDetectorParams as cfg
from streakiller.config.schema import OutputOptions
from streakiller.models.result import PipelineResult

logger = logging.getLogger(__name__)

# Okabe-Ito-inspired BGR palette chosen for stronger separation under
# red-green color vision deficiencies.
STAGE_OVERLAY_COLORS: list[tuple[int, int, int]] = [
    (233, 180, 86),   # sky blue
    (0, 159, 230),    # orange
    (115, 158, 0),    # bluish green
    (66, 228, 240),   # yellow
    (178, 114, 0),    # blue
    (0, 94, 213),     # vermillion
    (167, 121, 204),  # reddish purple
]

ANNOTATION_THICKNESS = 3
LEGEND_FONT_SCALE = 0.6
LEGEND_TEXT_THICKNESS = 1


@runtime_checkable
class OutputWriter(Protocol):
    def write(self, result: PipelineResult) -> None: ...


class LocalOutputWriter:
    """
    Writes pipeline results to the local filesystem.

    For each processed image, outputs are written to::

        output_dir / <source_stem> /
            detected_streaks.png
            streaks.csv
            filter_stage_overlays.png
            processing_results.json

    If ``options.save_intermediate_images`` is True, also writes::

        binary.png
        normalized_display.png
    """

    def __init__(self, output_dir: Path, options: OutputOptions | None = None) -> None:
        self._dir = Path(output_dir)
        self._options = options or OutputOptions()
        self._dir.mkdir(parents=True, exist_ok=True)
        self.fft_image_log = None
        self.fft_template_log = None
        self.fft_product_log = None

    def write(self, result: PipelineResult) -> None:
        stem = result.source_path.stem if result.source_path else "unknown"
        run_dir = self._dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_csv(result, run_dir)
        self._write_annotated_image(result, run_dir)
        self._write_overlay(result, run_dir)
        self._write_provenance(result, run_dir)

        if self._options.save_intermediate_images:
            self._write_intermediates(result, run_dir)

        if self._options.save_text_summary:
            self._write_text_summary(result, run_dir)

        # Mirror the final image over the vertical to match the orientation of the original FITS image, which is flipped by the Hough transform
        original_plot = PILImage.open(run_dir / "detected_streaks.png")
        mirrored_plot = original_plot.transpose(PILImage.FLIP_TOP_BOTTOM)
        mirrored_plot.save(run_dir / "final_results_hough.png")

        logger.info("Results written to %s", run_dir)

    # pca_fft_write() was intended to be an equivalent to write() for the pca_fft method, but is unused due to object attribute (self) conflicts when called in streak_pipeline.py
    '''
    def pca_fft_write(self, result: PipelineResult, normalized_img, streaks, output_dir) -> None:
        stem = result.source_path.stem if result.source_path else "unknown"
        if output_dir is not None:
            run_dir = Path(output_dir) / stem
        else:
            run_dir = Path(cfg.output_dir) / stem
        run_dir.mkdir(parents=True, exist_ok=True)

        self._REJECTION_COLORS(self)
        self.display_diagnostics_plot_1(self, normalized_img, run_dir)
        self.display_diagnostics_plot_2(self, normalized_img, streaks, run_dir)
        # if you want to see the Fourier components
        #self._calculate_fft_intermediates(self, normalized_img, run_dir)
        #self.display_fourier_correlation_steps(self, normalized_img, run_dir)

        self.display_nearest_neighbour_histogram(self, run_dir)
        self.display_line_profiles(self, normalized_img, streaks, run_dir)
        self.display_final_results(self, normalized_img, streaks, run_dir)

        if self._options.save_text_summary:
            self._write_text_summary_pca_fft(result, run_dir)

        logger.info("Results written to %s", run_dir)
    '''
    # ------------------------------------------------------------------ #
    # Private writers                                                      #
    # ------------------------------------------------------------------ #

    def _write_csv(self, result: PipelineResult, run_dir: Path) -> None:
        snr_by_index = {s.streak_index: s for s in result.snr_estimates}
        path = run_dir / "streaks.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "label", "x1", "y1", "x2", "y2",
                "midpoint_x", "midpoint_y",
                "snr", "signal_adu", "noise_adu",
            ])
            for i, line in enumerate(result.detected_lines):
                x1, y1, x2, y2 = line[0]
                snr = snr_by_index.get(i)
                writer.writerow([
                    str(i + 1), x1, y1, x2, y2,
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    f"{snr.snr:.2f}" if snr and snr.is_valid else "",
                    f"{snr.signal:.1f}" if snr and snr.is_valid else "",
                    f"{snr.noise:.2f}" if snr and snr.is_valid else "",
                ])
        logger.debug("Wrote %s", path)

    def _write_annotated_image(self, result: PipelineResult, run_dir: Path) -> None:
        if result.normalized_display is None:
            return
        display = result.normalized_display.copy()
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        for i, line in enumerate(result.detected_lines):
            x1, y1, x2, y2 = line[0]

            cv2.line(
                display,
                (x1, y1), # endpoint 1
                (x2, y2), # endpoint 2
                (0, 255, 0), # color
                ANNOTATION_THICKNESS,
                cv2.LINE_AA,
            )
            #xmin, xmax = min(x1, x2), max(x1, x2)
            #ymin, ymax = min(y1, y2), max(y1, y2)
            # green bounding box
            #cv2.rectangle(
            #    display,
            #    (xmin, ymin),
            #    (xmax, ymax),
            #    (0, 255, 0),
            #    ANNOTATION_THICKNESS,
            #)

            # yellow number
            #cv2.putText(
            #    display, str(i + 1), (xmin, ymin - 10),
            #    cv2.FONT_HERSHEY_SIMPLEX,
            #    0.45,
            #    (0, 255, 255),
            #    ANNOTATION_THICKNESS,
            #    cv2.LINE_AA,
            #)

        path = run_dir / "detected_streaks.png"
        cv2.imwrite(str(path), display)
        logger.debug("Wrote %s", path)

    def _write_overlay(self, result: PipelineResult, run_dir: Path) -> None:
        if not result.filter_snapshots or result.binary_image is None:
            return
        base = result.binary_image.copy()
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        stages = [("initial_detected", result.initial_detected_lines)] + [
            (s.stage_name, s.lines) for s in result.filter_snapshots
        ] + [("final", result.detected_lines)]
        overlay = _draw_filter_stage_overlays(base, stages)

        path = run_dir / "filter_stage_overlays.png"
        cv2.imwrite(str(path), overlay)
        logger.debug("Wrote %s", path)

    def _write_provenance(self, result: PipelineResult, run_dir: Path) -> None:
        if result.provenance is None:
            return
        snr_records = [
            {
                "streak_index": s.streak_index,
                "snr": None if not s.is_valid else round(s.snr, 3),
                "signal_adu": None if not s.is_valid else round(s.signal, 2),
                "noise_adu": None if not s.is_valid else round(s.noise, 3),
                "n_on_pixels": s.n_on_pixels,
                "n_off_pixels": s.n_off_pixels,
            }
            for s in result.snr_estimates
        ]
        record: dict = {
            "source_file": str(result.source_path),
            "streak_count": result.streak_count,
            "error": result.error,
            **{k: v for k, v in vars(result.provenance).items()},
            "snr_estimates": snr_records,
        }
        path = run_dir / "processing_results.json"
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        logger.debug("Wrote %s", path)

    def _write_text_summary(self, result: PipelineResult, run_dir: Path) -> None:
        p = result.provenance
        source = result.source_path.name if result.source_path else "unknown"
        width = 60

        def divider(char="─"):
            return char * width

        def row(label: str, value: str) -> str:
            return f"  {label:<28}{value}"

        lines = [
            divider("═"),
            "  STREAK DETECTION SUMMARY".center(width),
            divider("═"),
            "",
        ]

        # Source file
        lines += [
            row("File:", source),
            row("Processed:", p.processing_start_utc[:19].replace("T", "  ") if p else "—"),
            row("Duration:", _format_duration(p) if p else "—"),
            row("Software version:", p.software_version if p else "—"),
            "",
            divider(),
            "  RESULT",
            divider(),
            "",
            row("Streaks detected:", str(result.streak_count)),
            row("Status:", "SUCCESS" if result.succeeded else f"ERROR — {result.error}"),
            "",
        ]

        # Per-streak table
        if result.streak_count > 0:
            snr_by_index = {s.streak_index: s for s in result.snr_estimates}
            lines += [
                divider(),
                "  DETECTED STREAKS",
                divider(),
                "",
                f"  {'#':<5} {'Start (x,y)':<18} {'End (x,y)':<18} {'Length (px)':<12} {'SNR':<8}",
                f"  {'─'*4} {'─'*16:<18} {'─'*16:<18} {'─'*10:<12} {'─'*6:<8}", # table header
            ]
            for i, line in enumerate(result.detected_lines):
                x1, y1, x2, y2 = line[0]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                snr = snr_by_index.get(i)
                snr_str = f"{snr.snr:>6.1f}" if snr and snr.is_valid else "   N/A"
                lines.append(
                    f"  {i+1:<5} ({x1:>5}, {y1:>5})     ({x2:>5}, {y2:>5})     {length:>8.1f}     {snr_str}"
                )
            lines.append("")

        # Pipeline settings
        if p:
            lines += [
                divider(),
                "  PIPELINE SETTINGS",
                divider(),
                "",
                row("Background method:", p.background_method_used),
                row("Min line length:", f"{p.min_line_length_used:.1f} px"),
                row("Hough threshold:", str(p.hough_threshold_used)),
                "",
            ]

            # Filter stage counts
            if p.stage_line_counts:
                lines += [
                    divider(),
                    "  FILTER STAGES",
                    divider(),
                    "",
                ]
                prev = None
                for stage, count in p.stage_line_counts.items():
                    if prev is not None:
                        removed = prev - count
                        tag = f"(−{removed})" if removed > 0 else "(no change)"
                    else:
                        tag = "(raw detection)"
                    lines.append(row(f"{stage}:", f"{count:>4}  {tag}"))
                    prev = count
                lines.append("")

        lines += [divider("═"), ""]

        path = run_dir / "processing_results_hough.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("Wrote %s", path)

    def _write_intermediates(self, result: PipelineResult, run_dir: Path) -> None:
        if result.binary_image is not None:
            cv2.imwrite(str(run_dir / "binary.png"), result.binary_image)
        if result.normalized_display is not None:
            cv2.imwrite(str(run_dir / "normalized_display.png"), result.normalized_display)


    # ------------------------------------------------------------------
    # PCA FFT 5.  DIAGNOSTIC PLOTTING MODULES
    # ------------------------------------------------------------------

    def _REJECTION_COLORS(self):
        """
        Master dictionary associating rejection reasons with plot colors.
        """
        self._REJECTION_COLORS = {
            'prominence': 'yellow',
            'overlap':    'blue',
            'too_close':  'magenta',
            'edge':       'red',
            'saturated':  'purple',
            'circular':   'orange',
            'too_wide':   'cyan',
    }


    def display_diagnostics_plot_1(self, normalized_img, run_dir: Path, img_log) -> None:
        """
        Plots the original image and illustrates exactly which features were rejected 
        from becoming the Master Template, and which one won.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # ── Panel 1: Original Image ──────────────────────────────────
        # Display with 1st/99th percentiles for good contrast
        axs[0].imshow(normalized_img, cmap='gray', origin='lower',
                        vmin=np.percentile(normalized_img, 1),
                        vmax=np.percentile(normalized_img, 99))

        legend_handles = []

        # Draw all the candidate features that were evaluated
        for f in self._all_features:
            reason = f['rejection']
            if reason is None:
                continue
            color  = self._REJECTION_COLORS.get(reason, 'white')
            coords = f['coords']
            
            label_text = f'Rejected: {reason}'
            
            handle = axs[0].plot(coords[:, 1], coords[:, 0], '.',
                                    markersize=1, color=color, alpha=0.4,
                                    label=label_text)[0]
            legend_handles.append(handle)

        # Draw the champion Master Template in green
        if self.template_coords is not None:
            h = axs[0].plot(self.template_coords[:, 1],
                            self.template_coords[:, 0],
                            'g.', markersize=1, alpha=0.4,
                            label='Selected template')[0]
            legend_handles.append(h)

        # Draw a bounding box around the chosen template for visibility
        if self.best_template_bbox is not None:
            y_start, y_end, x_start, x_end = self.best_template_bbox
            rect = patches.Rectangle(
                (x_start, y_start),
                x_end - x_start, y_end - y_start,
                linewidth=2, edgecolor='lime', facecolor='none',
                linestyle='--', label='Template box')
            axs[0].add_patch(rect)

        # Generate custom key symbols for the legend
        for reason, color in self._REJECTION_COLORS.items():
            label_text = f'Rejected: {reason}'
            axs[0].plot([], [], 's', color=color, label=label_text)
            
        # Deduplicate the legend! Without this, matplotlib will create like 500 legend entries.
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7, markerscale=4)

        axs[0].set_title("1. Original Image – Template Selection")

        # ── Panel 2: Chosen Template ─────────────────────────────────
        if self.best_template is not None:
            axs[1].imshow(self.best_template, cmap='gray', origin='lower')
            axs[1].set_title(
                f"2. Best Template  (angle: {self.best_template_angle:.2f}°)")
        else:
            axs[1].set_title("2. Template not created")

        path = run_dir / "template_selection.png"
        plt.tight_layout()
        plt.savefig(path)
        img_log.info("Wrote template_selection.png")
        logger.debug("Wrote %s", path)
        

    def display_diagnostics_plot_2(self, normalized_img, streaks, run_dir: Path, img_log) -> None:
        """
        Plots the Correlation Map (Heatmap) output by the FFT algorithm, alongside 
        the extracted peaks and nearest-neighbor linkages.
        """
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        if self.correlation_map is not None:
            # ── Panel 1: Nearest Neighbors ───────────────────────────────
            axs[0].imshow(self.correlation_map, cmap='viridis', origin='lower')
            if (self.detected_peaks_coords is not None and len(self.detected_peaks_coords) > 0):
                axs[0].plot(self.detected_peaks_coords[:, 0],
                            self.detected_peaks_coords[:, 1],
                            'r+', markersize=10, alpha=0.6,
                            label='Detected peaks')

                # Draw thin lines connecting every peak to its closest neighbor
                if hasattr(self, 'nn_indices') and self.nn_distances.size > 0:
                    plotted_nn = set()
                    for i, j in enumerate(self.nn_indices):
                        # Use a sorted tuple to ensure line A->B is treated the same as B->A (prevent duplicate drawing)
                        key = tuple(sorted((i, int(j))))
                        if key in plotted_nn:
                            continue
                        plotted_nn.add(key)
                        xi, yi = self.detected_peaks_coords[i]
                        xj, yj = self.detected_peaks_coords[int(j)]
                        axs[0].plot([xi, xj], [yi, yj], 'c-', lw=0.8, alpha=0.5)
                        
            axs[0].set_title("3. Correlation Map, Peaks & Nearest-Neighbour Links")
            axs[0].legend(loc='upper right', fontsize=8)

            # ── Panel 2: Detected Streaks ────────────────────────────────
            axs[1].imshow(self.correlation_map, cmap='viridis', origin='lower')
            for streak in streaks:
                (x1, y1), (x2, y2) = streak['endpoints']
                reason = streak['rejection']
                
                # In the final detection context, 'prominence' just means it wasn't the absolute brightest, 
                # but it is still fully accepted.
                if reason is None or reason == 'prominence':
                    color = 'lime' if reason is None else 'yellow'
                    label = 'Accepted'
                else:
                    color = self._REJECTION_COLORS.get(reason, 'red')
                    label = f'Rejected: {reason}'
                    
                axs[1].plot([x1, x2], [y1, y2], color=color, lw=1.5, alpha=0.8, label=label)
            
            # Deduplicate the legend
            handles, labels = axs[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[1].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)
            axs[1].set_title("4. Correlation Map & Final Detections")
            axs[1].set_xlim(0, normalized_img.shape[1])
            axs[1].set_ylim(0, normalized_img.shape[0])
        else:
            axs[0].set_title("3. Correlation map not calculated")
            axs[1].set_title("4. Correlation map not calculated")

        path = run_dir / "correlation_map.png"
        plt.tight_layout()
        plt.savefig(path)
        img_log.info("Wrote correlation_map.png")
        logger.debug("Wrote %s", path)


    def _calculate_fft_intermediates(self, normalized_img, img_log) -> None:
        """
        Computes the Fast Fourier Transform (FFT) magnitudes for the original image and the template.
        This is strictly used for the `display_fourier_correlation_steps` visualization plot.
        """
        img_log.info("Calculating FFT intermediates...")

        # Image FFT
        fft_img         = np.fft.fft2(normalized_img)
        fft_img_shifted = np.fft.fftshift(fft_img) # Shift low frequencies to the center of the image
        self.fft_image_log = np.log(np.abs(fft_img_shifted) + 1e-9)

        # Template FFT (padded to match image size)
        padded = np.zeros_like(normalized_img)
        th, tw = self.best_template.shape
        ih, iw = normalized_img.shape
        cy, cx = ih // 2, iw // 2
        padded[cy - th//2: cy + th - th//2, cx - tw//2: cx + tw - tw//2] = self.best_template

        fft_tpl         = np.fft.fft2(padded)
        fft_tpl_shifted = np.fft.fftshift(fft_tpl)
        self.fft_template_log = np.log(np.abs(fft_tpl_shifted) + 1e-9)

        # Spectral Product (Image * Conjugate of Template) - This represents the correlation in frequency space!
        fft_product         = fft_img * np.conj(fft_tpl)
        fft_product_shifted = np.fft.fftshift(fft_product)
        self.fft_product_log = np.log(np.abs(fft_product_shifted) + 1e-9)


    def display_fourier_correlation_steps(self, run_dir: Path, img_log) -> None:
        """Displays the internal spatial frequencies used by the FFT correlation algorithm."""
        if self.fft_image_log is None:
            return
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        axs[0].imshow(self.fft_image_log,    cmap='gray',   origin='lower')
        axs[0].set_title("A. FFT Magnitude of Image")
        axs[0].axis('off')
        axs[1].imshow(self.fft_template_log, cmap='gray',   origin='lower')
        axs[1].set_title("B. FFT Magnitude of Template")
        axs[1].axis('off')
        axs[2].imshow(self.fft_product_log,  cmap='inferno', origin='lower')
        axs[2].set_title("C. Spectral Product (Image × Template*)")
        axs[2].axis('off')

        path = run_dir / "fourier_correlation_steps.png"
        plt.tight_layout()
        plt.savefig(path)
        img_log.info("Wrote fourier_correlation_steps.png")
        logger.debug("Wrote %s", path)


    def display_nearest_neighbor_histogram(self, run_dir: Path, img_log) -> None:
        """Plots a histogram showing the distribution of distance between streaks."""
        if not hasattr(self, 'nn_distances') or self.nn_distances.size == 0:
            img_log.warning("No NN data to plot.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.nn_distances, bins=20, color='steelblue', edgecolor='white')
        ax.set_xlabel("Distance to nearest-neighbor peak (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Nearest-Neighbour Distance Distribution")
        plt.tight_layout()

        path = run_dir / "nearest_neighbor_histogram.png"
        plt.savefig(path)
        img_log.info("Wrote nearest_neighbor_histogram.png")
        logger.debug("Wrote %s", path)


    def display_line_profiles(self, normalized_img, streaks, run_dir: Path, img_log) -> None:
        """
        Draws an exact 1D line slice straight down the mathematical center of the primary streak,
        plotting its brightness across the image vs. its correlation confidence score.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        if not streaks:
            for ax in axs:
                ax.set_title("No streaks detected")
            plt.tight_layout()
            path = run_dir / "line_profiles.png"
            plt.savefig(path)
            logger.debug("Wrote %s", path)
            return

        (x1, y1), (x2, y2) = streaks[0]['endpoints']
        
        # Linearly interpolate points between the start and end of the streak
        t = np.linspace(0, 1, cfg.profile_length)
        x_idx = x1 + (x2 - x1) * t
        y_idx = y1 + (y2 - y1) * t
        coords = np.vstack((y_idx, x_idx))

        # Map the coordinates back onto the heatmap and original image arrays to extract values
        if self.correlation_map is not None:
            cp = map_coordinates(self.correlation_map, coords, order=1)
            axs[0].plot(t, cp, 'c-')
            axs[0].set_title("5. Correlation Profile Along Streak")
            axs[0].set_xlabel("Normalised distance")
            axs[0].set_ylabel("Correlation value")

        bp = map_coordinates(normalized_img, coords, order=1)
        axs[1].plot(t, bp, 'm-')
        axs[1].set_title("6. Base Image Profile Along Streak")
        axs[1].set_xlabel("Normalised distance")
        axs[1].set_ylabel("Pixel intensity")

        plt.tight_layout()
        path = run_dir / "line_profiles.png"
        plt.savefig(path)
        img_log.info("Wrote line_profiles.png")
        logger.debug("Wrote %s", path)


    def display_final_results(self, normalized_img, streaks, run_dir: Path, img_log) -> None:
        """
        The final master plot overlaying every accepted and rejected streak line directly 
        onto the original FITS image.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(normalized_img, cmap='gray', origin='lower',
                    vmin=np.percentile(normalized_img, 1),
                    vmax=np.percentile(normalized_img, 99))
        
        self.accepted_count = 0
        '''
        for streak in streaks:
            (x1, y1), (x2, y2) = streak['endpoints']
            reason = streak['rejection']

            # In the final detection context, 'prominence' just means it wasn't the absolute brightest, 
            # but it is still fully accepted.
            if reason is None or reason == 'prominence':
                self.accepted_count += 1
                self.accepted_streaks.append(streak)
                if reason is None:
                    color = 'lime'
                    label = 'Best'
                else:
                    color = 'yellow'
                    label = 'Accepted'
                alpha, lw = 0.9, 2.0 
            else:
                color, alpha, lw = self._REJECTION_COLORS.get(reason, 'red'), 0.4, 1.0 
                label = f'Rejected: {reason}'
            
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, label=label)
        '''
        # First pass — rejected streaks (drawn underneath)
        for streak in streaks:
            reason = streak['rejection']
            if reason is None or reason == 'prominence':
                continue
            (x1, y1), (x2, y2) = streak['endpoints']
            color = self._REJECTION_COLORS.get(reason, 'red')
            ax.plot([x1, x2], [y1, y2], color=color, lw=1.0, alpha=0.4,
                    label=f'Rejected: {reason}')

        # Second pass — accepted streaks (drawn on top)
        for streak in streaks:
            reason = streak['rejection']
            if reason is None or reason == 'prominence':
                self.accepted_count += 1
                self.accepted_streaks.append(streak)
                (x1, y1), (x2, y2) = streak['endpoints']
                color = 'lime' if reason is None else 'yellow'
                label = 'Accepted' if reason is None else 'Accepted'
                ax.plot([x1, x2], [y1, y2], color=color, lw=2.0, alpha=0.9, label=label)
        
        title = f"Final Detections ({self.accepted_count} valid, {len(streaks) - self.accepted_count} rejected)"
        
        # Display the length used for calculations in the title dynamically
        length_to_display = self.theoretical_streak_length_px if (self.use_exposure_length and self.theoretical_streak_length_px) else self.expected_streak_length_px
        
        if length_to_display:
            title += (f"\n(Expected streak length ≈ "
                        f"{length_to_display:.0f} px"
                        f" | Exposure = {self.exposure_time} s)")
                        
        ax.set_title(title)
        ax.set_xlim(0, normalized_img.shape[1])
        ax.set_ylim(0, normalized_img.shape[0])
        
        # Deduplicate Legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        plt.tight_layout()
        path = run_dir / "final_results_pca_fft.png"
        plt.savefig(path)
        img_log.info("Wrote final_results_pca_fft.png")
        logger.debug("Wrote %s", path)


    def _write_text_summary_pca_fft(self, result: PipelineResult, run_dir: Path, streaks) -> None:
        p = result.provenance
        source = result.source_path.name if result.source_path else "unknown"
        width = 60

        def divider(char="─"):
            return char * width

        def row(label: str, value: str) -> str:
            return f"  {label:<28}{value}"

        lines = [
            divider("═"),
            "  STREAK DETECTION SUMMARY".center(width),
            divider("═"),
            "",
        ]

        # Source file
        lines += [
            row("File:", source),
            row("Processed:", p.processing_start_utc[:19].replace("T", "  ") if p else "—"),
            row("Duration:", _format_duration(p) if p else "—"),
            row("Software version:", p.software_version if p else "—"),
            "",
            divider(),
            "  RESULT",
            divider(),
            "",
            row("Streaks detected:", str(self.accepted_count)),
            row("Status:", "SUCCESS" if result.succeeded else f"ERROR — {result.error}"),
            "",
        ]
        
        # Per-streak table
        if self.accepted_count is not None:
            #snr_by_index = {s.streak_index: s for s in result.snr_estimates} # fix this
            lines += [
                divider(),
                "  DETECTED STREAKS",
                divider(),
                "",
                f"  {'#':<5} {'Start (x,y)':<18} {'End (x,y)':<18} {'Length (px)':<12} {'SNR':<8}",
                f"  {'─'*4} {'─'*16:<18} {'─'*16:<18} {'─'*10:<12} {'─'*6:<8}", # table header
            ]

            for i, line in enumerate(self.accepted_streaks):
                x1, y1, x2, y2 = int(line['endpoints'][0][0]), int(line['endpoints'][0][1]), int(line['endpoints'][1][0]), int(line['endpoints'][1][1])
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                #snr = snr_by_index.get(i)
                snr_str='test'
                #snr_str = f"{snr.snr:>6.1f}" if snr and snr.is_valid else "   N/A"
                lines.append(
                    f"  {i+1:<5} ({x1:>5}, {y1:>5})     ({x2:>5}, {y2:>5})     {length:>8.1f}     {snr_str}"
                )
            lines.append("")
        
        # Pipeline settings
        if p:
            lines += [
                divider(),
                "  PIPELINE SETTINGS",
                divider(),
                "",
                '''
                row("Background method:", p.background_method_used),
                row("Min line length:", f"{p.min_line_length_used:.1f} px"),
                row("Hough threshold:", str(p.hough_threshold_used)),
                "",
                '''
            ]
            
            # Filter stage counts
            #if p.stage_line_counts:
            lines += [
                divider(),
                "  FILTER STAGES",
                divider(),
                "",
            ]

            if streaks is not None:
                lines.append(row(f"INITIAL:", f"{len(streaks):>4}  (raw detection)"))
                # edge
                remaining = len(streaks) - self.edge_rejected
                removed = f"(−{self.edge_rejected})"
                lines.append(row(f"EDGE:", f"{remaining:>4}  {removed}"))

                # proximity
                remaining = remaining - self.proximity_rejected
                removed = f"(−{self.proximity_rejected})"
                lines.append(row(f"PROXIMITY:", f"{remaining:>4}  {removed}"))

                # prominence (should always result in 1 remaining streak, the brightest one)
                remaining = remaining - self.prominence_rejected
                removed = f"(−{self.prominence_rejected})"
                lines.append(row(f"PROMINENCE:", f"{remaining:>4}  {removed}"))
                lines.append("")
            
        lines += [divider("═"), ""]
        
        path = run_dir / "processing_results_pca_fft.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("Wrote %s", path)


# ------------------------------------------------------------------ #
# Private helpers                                                     #
# ------------------------------------------------------------------ #

def _format_duration(provenance) -> str:
    from datetime import datetime, timezone
    try:
        start = datetime.fromisoformat(provenance.processing_start_utc)
        end = datetime.fromisoformat(provenance.processing_end_utc)
        seconds = (end - start).total_seconds()
        if seconds < 60:
            return f"{seconds:.1f}s"
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    except Exception:
        return "—"


# ------------------------------------------------------------------ #
# Visualisation helper                                                #
# ------------------------------------------------------------------ #

def _draw_filter_stage_overlays(
    base_image: np.ndarray,
    stages: list[tuple[str, np.ndarray]],
    thickness: int = ANNOTATION_THICKNESS,
) -> np.ndarray:
    out = base_image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    n = max(1, len(stages))

    for idx, (name, lines) in enumerate(stages):
        if lines is None or len(lines) == 0:
            continue
        color = STAGE_OVERLAY_COLORS[idx % len(STAGE_OVERLAY_COLORS)]
        overlay = out.copy()
        for line in lines:
            l = np.asarray(line).reshape(-1)
            if l.size < 4:
                continue
            cv2.line(
                overlay,
                (int(l[0]), int(l[1])),
                (int(l[2]), int(l[3])),
                color,
                thickness,
                cv2.LINE_AA,
            )
        alpha = 0.35 + 0.55 * (idx / max(1, n - 1))
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    # Legend
    pad = 50
    lx, ly = pad, pad
    box_w, box_h = 350, 35 * len(stages) + 12
    legend_bg = out.copy()
    cv2.rectangle(legend_bg, (lx - 4, ly - 4), (lx + box_w, ly + box_h), (0, 0, 0), -1)
    cv2.addWeighted(legend_bg, 0.45, out, 0.55, 0, out)

    for idx, (name, lines) in enumerate(stages):
        color = STAGE_OVERLAY_COLORS[idx % len(STAGE_OVERLAY_COLORS)]
        ty = ly + 6 + idx * 35
        cv2.rectangle(out, (lx, ty), (lx + 20, ty + 20), color, -1)
        count = len(lines) if lines is not None and hasattr(lines, "__len__") else 0
        cv2.putText(
            out,
            f"{name}: {count}",
            (lx + 30, ty + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            LEGEND_FONT_SCALE,
            (255, 255, 255),
            LEGEND_TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    return out
