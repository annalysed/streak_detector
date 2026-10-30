import matplotlib.pyplot as plt
import matplotlib.patches as patches

def bounding_box(x1, y1, x2, y2, padding):
    """Compute axis-aligned bounding box around a line segment with optional padding."""
    bx1 = min(x1, x2) - padding
    by1 = min(y1, y2) - padding
    bx2 = max(x1, x2) + padding
    by2 = max(y1, y2) + padding
    return (bx1, by1, bx2, by2)


def boxes_intersect(a, b):
    """Return True if two bounding boxes intersect."""
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b
    return (x1a <= x2b and x2a >= x1b and
            y1a <= y2b and y2a >= y1b)


# --- Define two line segments as (x1, y1, x2, y2) ---
streaks = {
     'bbox': (1718, 1817, 2572, 2895)}, {
     'bbox': (1700, 1800, 2550, 2870)}


proximity_margin = 1

for i, streak in enumerate(streaks):
    print(f'i={i}')
    y1, y2, x1, x2 = streak['bbox']
    line1 = (x1, y1, x2, y2)

    # Compute bounding box with proximity_margin
    box1 = bounding_box(*line1, padding=proximity_margin)

    # 2. Proximity Rejection (Streaks too close to each other)
    # Checks the current streak against all *previously accepted* (brighter) streaks. Required for clean Pill Photometry.
    for j in range(i):
        print(f'j={j}')
        prev_streak = streaks[j]
        #if prev_streak['rejection'] is None:
        py1, py2, px1, px2 = prev_streak['bbox']
        line2 = (px1, py1, px2, py2)

        # Compute bounding box with proximity_margin
        box2 = bounding_box(*line2, padding=proximity_margin)
        # change streak['bbox'] to actually hold the bbox
        # problem is, bounding_box is (x1, y1, x2, y2) and bbox is (y1, y2, x1, x2)
        #prev_streak['bbox'] = box2

        # Check intersection
        intersects = boxes_intersect(box1, box2)
        print(f"Box 1: {box1}")
        print(f"Box 2: {box2}")
        print(f"Bounding boxes intersect: {intersects}")

    
        # --- Plot ---
        fig, ax = plt.subplots()

        # Draw line segments
        ax.plot([line1[0], line1[2]], [line1[1], line1[3]], 'b-', linewidth=2, label='Line 1')
        ax.plot([line2[0], line2[2]], [line2[1], line2[3]], 'r-', linewidth=2, label='Line 2')

        # Draw bounding boxes
        x1a, y1a, x2a, y2a = box1
        ax.add_patch(patches.Rectangle(
            (x1a, y1a), x2a - x1a, y2a - y1a,
            linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.2, label='Box 1'
        ))

        x1b, y1b, x2b, y2b = box2
        ax.add_patch(patches.Rectangle(
            (x1b, y1b), x2b - x1b, y2b - y1b,
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.2, label='Box 2'
        ))

        # Draw intersection region if it exists
        if intersects:
            ix1 = max(x1a, x1b)
            iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b)
            iy2 = min(y2a, y2b)
            ax.add_patch(patches.Rectangle(
                (ix1, iy1), ix2 - ix1, iy2 - iy1,
                linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.4, label='Intersection'
            ))

        # Formatting
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True)
        ax.set_title(f"Bounding Box Intersection: {'Yes' if intersects else 'No'}")
        plt.show()