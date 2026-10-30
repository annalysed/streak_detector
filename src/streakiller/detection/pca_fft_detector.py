"""
Streak detection via PCA template extraction + FFT cross-correlation.

Algorithm:
1. Threshold the image to isolate bright objects (99th percentile by default).
2. Label connected components and measure their shapes via PCA covariance.
3. Select the most elongated, non-edge, thin component as the Template.
4. Cross-correlate the image with the template using FFT convolution.
5. Extract peaks from the correlation map as streak centres.
6. Compute streak endpoints from each centre using the template angle and length.

Ported from FourierStreakDetector (Streak_Detector.py) and adapted to the streakiller interface.
"""
from __future__ import annotations
import logging
import numpy as np

from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from scipy.ndimage import binary_closing, find_objects, label
from scipy.signal import fftconvolve
from skimage.feature import peak_local_max
from sklearn.neighbors import BallTree

from streakiller.models.fits_image import FitsImage
from streakiller.config.schema import FftDetectorParams as cfg

logger = logging.getLogger(__name__)


class PCAFFTDetector:
    """
    A class to automatically detect and extract satellite or orbital debris streaks in astronomical FITS images.
    
    The pipeline works by:
    1. Isolating the brightest pixels to find the longest, most prominent streak in the image.
    2. Using Principal Component Analysis (PCA) on the streak's shape to determine its exact angle, length, and width.
    3. Extracting that streak to use as a "Template".
    4. Using Fast Fourier Transform (FFT) cross-correlation to find all other regions in the image that match the template.
    5. Filtering out false positives (too close to edge, too close to another streak, or too circular).

    Unlike HoughLinesP, this method does not require a pre-computed binary mask:
    it derives its own mask from the raw image.  The *binary* argument accepted by
    detect() exists only for interface compatibility and is intentionally ignored.
    """

    # __init__ ported directly from elizabeth's 
    def __init__(self, min_box_separation=0):

        """
        Parameters
        ----------

        min_box_separation : float
            Minimum fractional overlap allowed between the template bounding
            box and any *other* streak bounding box (0 = no overlap required).
        """
        self.min_box_separation = min_box_separation

        # Core image data
        #self.image_data = None
        self.header = None
        self.streaks = []

        # Template attributes
        self.best_template = None
        self.best_template_angle = 0
        self.best_template_bbox = None   # (y_start, y_end, x_start, x_end)
        self.template_coords = None

        # Rejection bookkeeping and feature storage
        self._rejection_map = {}
        self._all_features = []          # List of dictionaries holding metadata for every detected object

        # Intermediates for FFT/Correlation math and diagnostics
        self.correlation_map = None
        self.detected_peaks_coords = None
        self.fft_image_log = None
        self.fft_template_log = None
        self.fft_product_log = None

        # Telemetry / Physical attributes extracted from FITS
        #self.exposure_time = None      # Exposure time in seconds
        #self.pixel_scale_arcsec = None   # Plate scale in arcseconds per pixel
        self.expected_streak_length_px = None # Length derived mathematically from the template
        
        # --- variables for future exposure-based length calculations ---
        self.theoretical_streak_length_px = None 
        self.use_exposure_length = False # Keep False to preserve the template-driven behavior by default
    
 # ------------------------------------------------------------------
    # 1.  LOADING & PREPROCESSING
    # ------------------------------------------------------------------
        
    def _parse_exposure_info(self, image: FitsImage, img_log):
        """
        Reads the exposure time and the pixel scale (arcseconds/pixel) from the FITS header. If standard keywords fail, it uses Astropy's World Coordinate System (WCS) to derive it mathematically.
        """
        pixel_scale_arcsec = None
        exposure_time = None
        h = image.raw_header
        
        # Look for standard exposure time keys
        for key in ('EXPTIME', 'EXPOSURE', 'EXPTIMED', 'EXP_TIME'):
            if key in h:
                exposure_time = float(h[key])
                img_log.info(f"  Exposure time: {exposure_time} s  (key={key})")
                break
        if exposure_time is None:
            img_log.warning("  WARNING: No exposure-time keyword found in header.")

        # Look for standard plate scale keys (CD matrices or CDELT keys)
        if 'CDELT2' in h:
            pixel_scale_arcsec = abs(float(h['CDELT2'])) * 3600.0
        elif 'CD2_2' in h:
            pixel_scale_arcsec = abs(float(h['CD2_2'])) * 3600.0
        else:
            # Fallback: If the image is plate-solved, Astropy can calculate the scale using WCS
            try:
                w = WCS(h)
                if w.has_celestial:
                    # proj_plane_pixel_scales returns degrees/pixel. Multiply by 3600 to convert to arcsec/px
                    scales = proj_plane_pixel_scales(w) * 3600.0
                    pixel_scale_arcsec = float(np.mean(scales))
            except Exception as e:
                img_log.warning(f"  WARNING: Astropy WCS could not extract pixel scale: {e}")

        if pixel_scale_arcsec is not None:
            img_log.info(f"  Pixel scale: {pixel_scale_arcsec:.4f} arcsec/px")
        return exposure_time, pixel_scale_arcsec
    

    def _create_template_from_image(self, normalized_img, img_log):
        """
        This function identifies the best streak in the image to use as a template.
        It converts the image to a binary mask, measures the mathematical shape (covariance) of every 
        bright point, filters out the bad ones, and saves the best one.
        """
        if normalized_img is None:
            img_log.error("Image data not loaded.")
            raise RuntimeError("Image data not loaded.")

        img_log.info("Creating binary mask to find best streak template...")
        
        # Standard thresholding (like Otsu) assumes a 50/50 mix of light and dark pixels. 
        # Astronomical images are 99% dark sky, so Otsu fails. 
        # Instead, we aggressively threshold everything below the 99th percentile to 0 (black). 
        # This perfectly isolates the 1% of bright signal (stars, streaks, galaxies).
        # We sample every 4th pixel ([::4, ::4]) to make this percentile math run 16x faster!
        thresh = np.percentile(normalized_img[::4, ::4], cfg.percentile_threshold)
        binary_mask = normalized_img > thresh
        
        # 'binary_closing' acts like a morphological glue. It fills in tiny 1-pixel gaps in our streaks 
        # so that a single streak doesn't accidentally get counted as two separate pieces.
        binary_mask = binary_closing(binary_mask, structure=np.ones((3,3)))
        
        # Label assigns a unique integer ID to every connected component of bright pixels
        labeled_image, num_features = label(binary_mask)

        if num_features == 0:
            img_log.error("No features found in binary mask to create template.")
            raise ValueError("No features found in binary mask to create template.")

        features = []

        # `find_objects` is a massive optimization. It returns the bounding box for every feature instantly, 
        # rather than forcing numpy to scan the entire 16-megapixel image thousands of times.
        slices = find_objects(labeled_image)

        # ---- Measure the physical characteristics of every object ----
        for i, slc in enumerate(slices):
            if slc is None:
                continue
                
            label_id = i + 1
            
            # Extract just the tiny stamp of the feature to save memory and time
            sub_labeled = labeled_image[slc]
            
            # Find the local (y, x) coordinates of every pixel that belongs to this specific feature
            local_coords = np.argwhere(sub_labeled == label_id)
            
            # Throw away tiny specks of noise
            if len(local_coords) < cfg.min_template_area:
                continue
                
            # Shift the local coordinates back to their real positions in the global image
            y_start, x_start = slc[0].start, slc[1].start
            coords = local_coords + np.array([y_start, x_start])

            # --- PRINCIPAL COMPONENT ANALYSIS (PCA) ON FEATURE SHAPE ---
            # By calculating the spatial covariance matrix of the pixel coordinates, we can mathematically 
            # determine exactly how long, how wide, and at what angle the object is sitting at!
            cov = np.cov(coords.T)
            if np.isnan(cov).any():
                continue
                
            # Eigenvalues tell us the variance (size) along the major and minor axes.
            # eigvals[1] is the length variance. eigvals[0] is the width variance.
            # Eigenvectors tell us the orientation (angle) of those axes.
            eigvals, eigvecs = np.linalg.eigh(cov)
            if eigvals[0] <= 1e-6: # Prevent division by zero for 1-pixel-wide artifacts
                continue

            # Elongation is the ratio of Length to Width
            elongation = np.sqrt(eigvals[1]) / np.sqrt(eigvals[0])

            # Calculate a basic bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = (y_min, y_max, x_min, x_max)

            # Measure how bright the feature is in the original un-labeled image
            # Sampling the entire bbox can include other features if bounding boxes overlap, so we calculate these from only the pixel values of the objects
            pixel_values = normalized_img[coords[:, 0], coords[:, 1]]
            mean_intensity = pixel_values.mean()
            max_intensity  = pixel_values.max()

            # Save all this math for later filtering and visualization
            features.append({
                'label':        label_id,
                'coords':       coords,
                'elongation':   elongation,
                'eigvals':      eigvals,
                'eigvecs':      eigvecs,
                'bbox':         bbox,
                'mean_int':     mean_intensity,
                'max_int':      max_intensity,
                'rejection':    None,
            })

        if not features:
            img_log.error("No valid features after initial filtering.")
            raise ValueError("No valid features after initial filtering.")


        # SORTING: We want the longest streak to act as our template.
        # We sort descending by PHYSICAL LENGTH (variance along the principal axis -> eigvals[1]).
        # (We don't sort by elongation, because a tiny 3x1 pixel noise artifact has an almost infinite elongation!)
        features.sort(key=lambda f: f['eigvals'][1], reverse=True)

        # ---- FILTER OUT BAD CANDIDATES ----
        valid_features = []
        ih, iw = normalized_img.shape


        for f in features:
            y_min, y_max, x_min, x_max = f['bbox']

            # 1. Edge Rejection: Don't build templates from streaks that bleed off the edge of the sensor
            if (x_min < cfg.template_edge_margin or x_max > iw - cfg.template_edge_margin or
                y_min < cfg.template_edge_margin or y_max > ih - cfg.template_edge_margin):
                img_log.info('EDGE')
                f['rejection'] = 'edge'
                continue

            # 2. Circularity Rejection: Streaks are lines. A perfect circle has an elongation of 1.0. 
            # If it is less than 3.0, it is probably just a slightly warped star.
            if f['elongation'] < 3.0: 
                img_log.info('CIRCULAR')
                f['rejection'] = 'circular'
                continue
                
            
            # 3. Width Rejection: Since sqrt(eigvals[0]) is essentially the 1-sigma width of the object in pixels, 
            # capping it at max_width_std ensures we strictly keep thin lines and get rid larger features.
            width_std = np.sqrt(f['eigvals'][0])
            if width_std > cfg.max_width_std: 
                f['rejection'] = 'too_wide'
                continue

            valid_features.append(f)

        if not valid_features:
            img_log.error("All features were rejected (edge, circular, etc.)")
            raise ValueError("All features were rejected (edge, circular, etc.")

        # The #1 feature remaining in the list is our champion Template!
        chosen = valid_features[0]
        cy1, cy2, cx1, cx2 = chosen['bbox']

        # Now, mark any features whose bounding box touches the Template as 'overlap'
        for f in valid_features[1:]:
            fy1, fy2, fx1, fx2 = f['bbox']
            # Rectangle intersection math
            inter_y = max(0, min(cy2, fy2) - max(cy1, fy1))
            inter_x = max(0, min(cx2, fx2) - max(cx1, fx1))
            if inter_y * inter_x > 0:
                f['rejection'] = 'overlap'

        # Any feature that survived all the filters, but wasn't the #1 longest one, gets tagged as 'prominence'.
        # These are usually perfectly good streaks, they just weren't selected to be the template!
        for f in valid_features[1:]:
            if f['rejection'] is None:
                f['rejection'] = 'prominence'

        self._all_features = features

        # ---- EXTRACT AND FINALIZE THE TEMPLATE IMAGE ----
        self.template_coords = chosen['coords']
        
        # Calculate the mathematical length of the streak in pixels using the variance
        primary_variance = chosen['eigvals'][1]
        self.expected_streak_length_px = np.sqrt(12 * primary_variance)
        img_log.info(f"Calculated universal streak length from template: {self.expected_streak_length_px:.1f} px")
        
        # Extract the angle in degrees using the principal eigenvector (y vs x)
        eigvecs = chosen['eigvecs']
        v = eigvecs[:, -1]
        self.best_template_angle = np.degrees(np.arctan2(v[0], v[1]))

        # Cut out the template from the original image, padding it slightly so it captures the background context
        y_min, y_max, x_min, x_max = chosen['bbox']
        y_start = max(0, y_min - cfg.template_padding)
        y_end   = min(ih, y_max + cfg.template_padding)
        x_start = max(0, x_min - cfg.template_padding)
        x_end   = min(iw, x_max + cfg.template_padding)
        self.best_template_bbox = (y_start, y_end, x_start, x_end)

        self.best_template = normalized_img[y_start:y_end, x_start:x_end].copy()

        # Mask out pixels not belonging to the chosen label
        template_labels = labeled_image[y_start:y_end, x_start:x_end]
        self.best_template[template_labels != chosen['label']] = 0.0

        if self.best_template.size == 0:
            img_log.error("Failed to create a valid template.")
            raise ValueError("Failed to create a valid template.")

        # Re-normalize just the template cutout to zero-mean. 
        # This is strictly required for the cross-correlation math in the next step to work properly.
        if np.std(self.best_template) > 1e-6:
            self.best_template = ((self.best_template - np.mean(self.best_template))
                                  / np.std(self.best_template))
        else:
            self.best_template = self.best_template - np.mean(self.best_template)

        img_log.info(f"Template created from streak at angle {self.best_template_angle:.2f}°  "
              f"(elongation={chosen['elongation']:.2f})")
    

    def detect_streaks_by_template(self, normalized_img, img_log, theoretical_streak_length_px, streak_length=None, use_exposure_length = False):
        """
        Uses the Template to scan the entire image and detect identical streaks.
        It uses Fast Fourier Transforms (FFT) to perform the scanning algorithm in a fraction of a second.
        """
        if normalized_img is None:
            raise RuntimeError("Image has not been loaded.")

        # Step 1: Guarantee the template exists
        #_create_template_from_image()

        # Step 2: Determine which length metric to use when drawing the final lines
        if streak_length is None:
            if use_exposure_length and theoretical_streak_length_px is not None:
                streak_length = theoretical_streak_length_px
            else:
                streak_length = (self.expected_streak_length_px
                                 if self.expected_streak_length_px
                                 else 50)
        
        img_log.info(f"Using streak_length={streak_length:.1f} px for endpoint calculation.")

        # Step 3: FFT Cross-Correlation
        # Standard cross-correlation slides the template pixel-by-pixel across the whole image. 
        # This takes $O(N \cdot M)$ time (often minutes!).
        # By convolving in the frequency domain using FFTs, the math drops to $O(N \log N)$ and finishes in < 1 second.
        # NOTE: Mathematical convolution flips the template. To simulate a true cross-correlation using convolution, 
        # we must physically flip the template 180 degrees first! `[::-1, ::-1]` accomplishes this.
        img_log.info("Performing template match (FFT-accelerated)...")
        flipped_template = self.best_template[::-1, ::-1]
        
        # `mode='same'` ensures the output correlation map is the exact same size as the input image
        self.correlation_map = fftconvolve(normalized_img, flipped_template, mode='same')
        img_log.info("Correlation map generated.")

        # Step 4: Extract the peaks
        # The correlation map contains bright "hotspots" wherever the template matched perfectly.
        # We set a threshold dynamically based on the median and standard deviation of the correlation map.
        mean_c  = np.mean(self.correlation_map)
        med_c   = np.median(self.correlation_map)
        std_c   = np.std(self.correlation_map)
        threshold_abs = med_c + cfg.threshold_sigma * std_c

        # `peak_local_max` finds the (y,x) coordinates of the hotspots
        peak_coords_yx = peak_local_max(
            self.correlation_map,
            min_distance=cfg.min_distance,
            threshold_abs=threshold_abs)

        if peak_coords_yx.size == 0:
            img_log.info("No peaks found.")
            self.detected_peaks_coords = np.empty((0, 2), dtype=int)
            return

        # Convert back to (x, y) coordinates for standard math
        self.detected_peaks_coords = peak_coords_yx[:, ::-1]
        img_log.info(f"Found {len(self.detected_peaks_coords)} peaks.")

        # Perform nearest-neighbor analysis for diagnostic purposes
        """
        Uses a highly optimized spatial BallTree (from scikit-learn) to calculate the distance 
        from every detected streak to its absolute closest neighbor.
        """
        #self._nearest_neighbour_analysis()
        if (self.detected_peaks_coords is None or
                len(self.detected_peaks_coords) < 2):
            self.nn_distances = np.array([])
            self.nn_indices   = np.array([], dtype=int)
            return

        coords_rad = np.deg2rad(self.detected_peaks_coords.astype(float))
        tree = BallTree(coords_rad, metric='euclidean')

        # Query the tree for k=2 nearest neighbors. 
        # (The closest point to point A is always point A itself, so we need k=2 to get the actual neighbor)
        distances, indices = tree.query(coords_rad, k=2)
        
        # Store the distance and index of the true nearest neighbor
        self.nn_distances = distances[:, 1]   
        self.nn_indices   = indices[:, 1]

        img_log.info(f"Nearest-neighbour stats: "
              f"min={self.nn_distances.min():.1f} px, "
              f"mean={self.nn_distances.mean():.1f} px, "
              f"max={self.nn_distances.max():.1f} px")

        # Step 5: Convert the center points into start and end points using trigonometry
        half_len  = streak_length / 2
        angle_rad = np.deg2rad(self.best_template_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        streaks = []

        for (x_c, y_c) in self.detected_peaks_coords:
            # dx, dy represent the horizontal and vertical distance from the center to the edge of the streak
            dx = half_len * cos_a
            dy = half_len * sin_a
            
            # Start Point (x1, y1) and End Point (x2, y2)
            x1, y1 = x_c - dx, y_c - dy
            x2, y2 = x_c + dx, y_c + dy
            
            # Save the peak correlation score so we can rank the streaks later
            corr_val = self.correlation_map[int(y_c), int(x_c)]
            
            # Store the final streak as a dictionary containing its geometric data
            streaks.append({
                'endpoints': ((x1, y1), (x2, y2)),
                'center': (x_c, y_c),
                'bbox': (min(y1, y2), max(y1, y2), min(x1, x2), max(x1, x2)),
                'corr_val': corr_val,
                'rejection': None # Default is accepted until the filter runs
            })
        return streaks
