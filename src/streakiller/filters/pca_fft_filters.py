"""
Streak detection via PCA template extraction + FFT cross-correlation.

Algorithm:
1. Threshold the image to isolate bright objects (99th percentile by default).
2. Label connected components and measure their shapes via PCA covariance.
3. Select the most elongated, non-edge, thin component as the Master Template.
4. Cross-correlate the image with the template using FFT convolution.
5. Extract peaks from the correlation map as streak centres.
6. Compute streak endpoints from each centre using the template angle and length.

Ported from FourierStreakDetector (Streak_Detector.py) and adapted to the streakiller interface.
"""
from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from streakiller.config.schema import FftDetectorParams as cfg

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 4.  FILTERING
# ------------------------------------------------------------------

def aa_bounding_box(x1, y1, x2, y2, padding):
    '''
    Computes an axis-aligned bounding box around a streak with optional padding.
    '''
    bx1 = min(x1, x2) - padding
    by1 = min(y1, y2) - padding
    bx2 = max(x1, x2) + padding
    by2 = max(y1, y2) + padding
    return (bx1, by1, bx2, by2)

def rotated_bounding_box(x1, y1, x2, y2, padding):
    '''
    Computes a streak-axis-aligned bounding box with padding.
    x1, y1, x2, y2: streak endpoints
    Returns the 4 vertices as a (4, 2) array with respect to the world space (the original xy axes).
    '''
    # Center, length, angle of streak so that we can treat the center of the streak as the origin in local space
    cx, cy = (x1 + x2)/2, (y1 + y2)/2
    length = np.hypot(x2 - x1, y2 - y1)

    # Half length and width so that we can make the bbox starting from the center
    half_len = length/2 + padding
    half_wid = padding

    # Bbox corners wrt streak's axes (length is +x)
    local_bbox = np.array([
        [-half_len, -half_wid],
        [ half_len, -half_wid],
        [ half_len,  half_wid],
        [-half_len,  half_wid],
    ])
    # Convert bbox to world space coords:
    # Apply rotation matrix to local space coords to rotate bbox
    # Add center coords in world space to put at original streak position
    angle  = np.arctan2(y2 - y1, x2 - x1)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a,  cos_a]])
    world_bbox = local_bbox @ R.T + np.array([cx, cy]) # the @ is for matrix multiplication
    return world_bbox

def aa_boxes_intersect(box1, box2):
    '''
    Returns True if two axis aligned bounding boxes intersect.
    '''
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    return (x1a <= x2b and x2a >= x1b and
            y1a <= y2b and y2a >= y1b)

def rotated_boxes_intersect(box1, box2):
    '''
    Determines if two boxes intersect by using the Separating Axis Theorem (SAT).
    SAT states that two convex shapes do not overlap if you can find a line onto which their projections do not intersect.
    box1, box2: (4, 2) arrays returned from rotated_bounding_box().
    Returns True if the boxes intersect.
    '''
    for box in [box1, box2]:
        n = len(box)
        for i in range(n):
            # Direction vector of one side of box (% n wraps around so the last corner connects back to the first)
            side = box[(i+1) % n] - box[i]
            # Axis vector, perpendicular to side vector
            axis = np.array([-side[1], side[0]])

            # Project both boxes onto axis
            proj_a = box1 @ axis
            proj_b = box2 @ axis

            # Intersection check
            if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                return False # do not intersect
    return True # do intersect

def _filter_detected_streaks(self, normalized_img, streaks):
    '''
    Runs through the final list of detected streaks and rejects ones that are invalid.
    Critically, it rejects streaks that are too close to each other, ensuring that downstream 
    tasks (like Pill Photometry) have a clean, uncontaminated background to measure against.
    '''
    ih, iw = normalized_img.shape

    # Sort the streaks by their correlation score descending, so we always process the highest confidence (brightest) streaks first.
    streaks.sort(key=lambda s: s['corr_val'], reverse=True)
    max_corr = streaks[0]['corr_val']

    # Loop through streaks from brightest to dimmest.
    for i, streak in enumerate(streaks):
        print(f'i={i}')
        if streak['rejection'] is None: # added this!
            y_min, y_max, x_min, x_max = streak['bbox']

            # 1. Edge Rejection (Streaks too close to the sensor edge)
            if (x_min < cfg.streak_edge_margin or x_max > iw - cfg.streak_edge_margin or
                y_min < cfg.streak_edge_margin or y_max > ih - cfg.streak_edge_margin):
                streak['rejection'] = 'edge'
                self.edge_rejected += 1
                continue

            # 2. Proximity Rejection (Streaks too close to each other)
            # Checks the current streak against all previously accepted (brighter) streaks.
            
            # Compute bounding box with proximity_margin
            #line1 = (x_min, y_min, x_max, y_max)
            #box1 = bounding_box(*line1, padding=cfg.proximity_margin)
            (x1, y1), (x2, y2) = streak['endpoints']
            box1 = rotated_bounding_box(x1, y1, x2, y2, padding=cfg.proximity_margin)
            
            for j, prev_streak in enumerate(streaks):
                print(f'j={j}')
                if i == j: continue # if they are equal, skip to the next j. makes it so that it doesnt check a streak against itself
                prev_streak = streaks[j]
                if streak['rejection'] is None and prev_streak['rejection'] is None:
                    #py_min, py_max, px_min, px_max = prev_streak['bbox']

                    # Compute bounding box with proximity_margin
                    (px1, py1), (px2, py2) = prev_streak['endpoints']
                    box2 = rotated_bounding_box(px1, py1, px2, py2, padding=cfg.proximity_margin)
                    # change streak['bbox'] to actually hold the bbox
                    # problem is, bounding_box is (x1, y1, x2, y2) and bbox is (y1, y2, x1, x2)
                    #prev_streak['bbox'] = box2
                    

                    # Check if the bounding boxes intersect. If so, the streaks are too close to safely measure flux.
                    intersects = rotated_boxes_intersect(box1, box2)

                    if intersects:
                        if streak['corr_val'] >= prev_streak['corr_val']:
                            prev_streak['rejection'] = 'too_close'
                        else:
                            streak['rejection'] = 'too_close'
                        self.proximity_rejected += 1
                        print(f'proximity rejected: {self.proximity_rejected}')


                    # Plot bbox_intersections figure
                    # fix directory: instead of plotting, save [x1, x2], [y1, y2] [px1, px2], [py1, py2] to array, pass array into output writer

                    # Draw streaks
                    plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2, label='Line 1')
                    plt.plot([px1, px2], [py1, py2], 'b-', linewidth=2, label='Line 2')

                    # Draw bounding boxes
                    #x1a, y1a, x2a, y2a = box1
                    #x1b, y1b, x2b, y2b = box2
                    #plt.gca().add_patch(patches.Polygon(box1, closed=True, linewidth=1, edgecolor='red', facecolor='red', alpha=0.01))
                    #plt.gca().add_patch(patches.Polygon(box2, closed=True, linewidth=1, edgecolor='red', facecolor='red', alpha=0.01))

                    # Draw intersection region if it exists
                    if intersects:
                        plt.gca().add_patch(patches.Polygon(box1, closed=True, linewidth=2, edgecolor='red', facecolor='red', alpha=0.1, label='Intersection'))
                        plt.gca().add_patch(patches.Polygon(box2, closed=True, linewidth=2, edgecolor='red', facecolor='red', alpha=0.1))
                    #    ix1 = max(x1a, x1b)
                    #    iy1 = max(y1a, y1b)
                    #    ix2 = min(x2a, x2b)
                    #    iy2 = min(y2a, y2b)
                    #    plt.gca().add_patch(patches.Rectangle(
                    #        (ix1, iy1), ix2 - ix1, iy2 - iy1,
                    #        linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.2, label='Intersection'
                    #    ))

        # 3. Prominence Rejection 
        # If the streak is less than 50% "confident" as the best streak in the image, reject it.
        #if streak['rejection'] is None and streak['corr_val'] < (max_corr * 0.5):
        #    streak['rejection'] = 'prominence'
        #    self.prominence_rejected += 1
    plt.savefig('bbox_intersections', dpi=300)
    plt.close()