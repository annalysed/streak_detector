import matplotlib.pyplot as plt
import matplotlib.patches as patches

streaks = {
     'bbox': (1718, 1817, 2572, 2895)}, {
     'bbox': (1713, 1812, 2567, 2890)}
     #'bbox': (0, 0, 4, 4)}, {
     #'bbox': (2, 2, 6, 6)}
proximity_margin = 1

for i, streak in enumerate(streaks):
        print(f'i={i}')
        y1, y2, x1, x2 = streak['bbox']
        proximity_margin = 20

        # Compute bounding box with proximity_margin
        x_min = min(x1, x2) - proximity_margin
        y_min = min(y1, y2) - proximity_margin
        x_max = max(x1, x2) + proximity_margin
        y_max = max(y1, y2) + proximity_margin

        
        # 2. Proximity Rejection (Streaks too close to each other)
        # Checks the current streak against all *previously accepted* (brighter) streaks. Required for clean Pill Photometry.
        for j in range(i):
            print(f'j={j}')
            prev_streak = streaks[j]
            #if prev_streak['rejection'] is None:
            y1, y2, x1, x2 = prev_streak['bbox']

            # Compute bounding box with proximity_margin
            xj_min = min(x1, x2) - proximity_margin
            yj_min = min(y1, y2) - proximity_margin
            xj_max = max(x1, x2) + proximity_margin
            yj_max = max(y1, y2) + proximity_margin

            # Each box defined as (x1, y1, x2, y2)
            a = (0, 0, 4, 4)
            b = (2, 2, 6, 6)

            # TO DO: FIGURE OUT WHICH COORDS HERE ARE MAX AND MIN AND GET PLOT TO WORK
            x1a, y1a, x2a, y2a = xj_max, xj_min, yj_max, yj_min
            x1b, y1b, x2b, y2b = x_max, x_min, y_max, y_min

            fig, ax = plt.subplots()

            # Draw box A
            ax.add_patch(patches.Rectangle(
                (x1a, y1a), x2a - x1a, y2a - y1a,
                linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='Box A'
            ))

            # Draw box B
            ax.add_patch(patches.Rectangle(
                (x1b, y1b), x2b - x1b, y2b - y1b,
                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Box B'
            ))

            # Check intersection and draw overlap region
            if x1a <= x2b and x2a >= x1b and y1a <= y2b and y2a >= y1b:
                ix1 = max(x1a, x1b)
                iy1 = max(y1a, y1b)
                ix2 = min(x2a, x2b)
                iy2 = min(y2a, y2b)
                ax.add_patch(patches.Rectangle(
                    (ix1, iy1), ix2 - ix1, iy2 - iy1,
                    linewidth=2, edgecolor='purple', facecolor='purple', alpha=0.5, label='Intersection'
                ))
                print("Boxes intersect")
            else:
                print("Boxes do NOT intersect")

            ax.set_aspect('equal')
            ax.legend()
            ax.grid(True)
            ax.set_title("Box Intersection")
            plt.show()

            #if x_max <= xj_min and x_min >= xj_max and y_max <= yj_min and y_min >= yj_max:
            #if inter_y * inter_x > 0:
            #    print(f'too_close')
            #    break
