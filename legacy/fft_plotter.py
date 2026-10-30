from __future__ import annotations
import logging

from streakiller.config.schema import FftDetectorParams as cfg

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import map_coordinates

class FftPlotter:
    # ------------------------------------------------------------------
    # 5.  DIAGNOSTIC PLOTTING MODULES
    # ------------------------------------------------------------------

    # Master dictionary associating rejection reasons with plot colors
    def _REJECTION_COLORS(self):
        self._REJECTION_COLORS = {
            'prominence': 'yellow',
            'overlap':    'blue',
            'too_close':  'magenta',
            'edge':       'red',
            'saturated':  'purple',
            'circular':   'orange',
            'too_wide':   'cyan',
    }

    def display_diagnostics_plot_1(self, normalized_img):
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

        plt.tight_layout()
        plt.savefig('Elizabeth_outputs/template_selection.png')

    def display_diagnostics_plot_2(self, normalized_img, streaks):
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
            
            # Deduplicate the legend!
            handles, labels = axs[1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs[1].legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)
            axs[1].set_title("4. Correlation Map & Final Detections")
            axs[1].set_xlim(0, normalized_img.shape[1])
            axs[1].set_ylim(0, normalized_img.shape[0])
        else:
            axs[0].set_title("3. Correlation map not calculated")
            axs[1].set_title("4. Correlation map not calculated")

        plt.tight_layout()
        plt.savefig('Elizabeth_outputs/correlation_map.png')

    def display_fourier_correlation_steps(self):
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
        plt.tight_layout()
        plt.savefig('Elizabeth_outputs/fourier_correlation_steps.png')

    def display_nearest_neighbour_histogram(self):
        """Plots a histogram showing the distribution of distance between streaks."""
        if not hasattr(self, 'nn_distances') or self.nn_distances.size == 0:
            print("No NN data to plot.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(self.nn_distances, bins=20, color='steelblue', edgecolor='white')
        ax.set_xlabel("Distance to nearest-neighbour peak (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Nearest-Neighbour Distance Distribution")
        plt.tight_layout()
        plt.savefig('Elizabeth_outputs/nearest_neighbour_histogram.png')

    def display_line_profiles(self, normalized_img, streaks):
        """
        Draws an exact 1D line slice straight down the mathematical center of the primary streak,
        plotting its brightness across the image vs. its correlation confidence score.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        if not streaks:
            for ax in axs:
                ax.set_title("No streaks detected")
            plt.tight_layout()
            plt.savefig('Elizabeth_outputs/line_profiles.png')
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
        plt.savefig('Elizabeth_outputs/coords_heatmap_original_img.png')

    def display_final_results(self, normalized_img, streaks):
        """
        The final master plot overlaying every accepted and rejected streak line directly 
        onto the original FITS image.
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(normalized_img, cmap='gray', origin='lower',
                    vmin=np.percentile(normalized_img, 1),
                    vmax=np.percentile(normalized_img, 99))
        
        accepted_count = 0
        for streak in streaks:
            (x1, y1), (x2, y2) = streak['endpoints']
            reason = streak['rejection']
            
            # Draw accepted streaks bold and green, and rejected streaks faint and color-coded
            if reason is None or reason == 'prominence':
                accepted_count += 1
                color = 'lime' if reason is None else 'yellow'
                alpha, lw = 0.9, 2.0 
                label = 'Accepted'
            else:
                color, alpha, lw = self._REJECTION_COLORS.get(reason, 'red'), 0.4, 1.0 
                label = f'Rejected: {reason}'
            
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, label=label)

        title = f"Final Detections ({accepted_count} valid, {len(streaks) - accepted_count} rejected)"
        
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
        plt.savefig('Elizabeth_outputs/final_results.png')
