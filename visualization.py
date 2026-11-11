"""OpenCV top-down normalized pitch renderer."""
import cv2
import numpy as np

def draw_pitch(players, ball=None, ball_observed=False, width=900, height=580,
               trail=None):
    margin = 35
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (45, 115, 65)
    x0, y0, x1, y1 = margin, margin, width-margin, height-margin
    white = (235, 235, 235)
    cv2.rectangle(canvas, (x0,y0), (x1,y1), white, 2)
    xm, ym = (x0+x1)//2, (y0+y1)//2
    cv2.line(canvas, (xm,y0), (xm,y1), white, 2)
    radius = int((y1-y0) * 0.12)
    cv2.circle(canvas, (xm,ym), radius, white, 2)
    cv2.circle(canvas, (xm,ym), 4, white, -1)
    # Approximate penalty and goal areas in normalized coordinates.
    for side in (0, 1):
        if side == 0:
            px0, px1 = x0, x0 + int((x1-x0)*0.16)
            gx0, gx1 = x0, x0 + int((x1-x0)*0.055)
        else:
            px0, px1 = x1-int((x1-x0)*0.16), x1
            gx0, gx1 = x1-int((x1-x0)*0.055), x1
        py0, py1 = y0+int((y1-y0)*0.20), y1-int((y1-y0)*0.20)
        gy0, gy1 = y0+int((y1-y0)*0.36), y1-int((y1-y0)*0.36)
        cv2.rectangle(canvas, (px0,py0), (px1,py1), white, 2)
        cv2.rectangle(canvas, (gx0,gy0), (gx1,gy1), white, 2)

    def pixel(pos):
        x, y = pos
        return (int(x0 + x*(x1-x0)), int(y0 + y*(y1-y0)))

    for p in players:
        if p.get("field_position") is None:
            continue
        pt = pixel(p["field_position"])
        cv2.circle(canvas, pt, 9, (40, 90, 245), -1)
        cv2.putText(canvas, str(p["id"]), (pt[0]+9, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
    if trail:
        pts = [pixel(p) for p in trail if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, (0, 220, 255), 2)
    if ball is not None:
        cv2.circle(canvas, pixel(ball), 7, (0, 220, 255) if ball_observed else (0,165,255), -1)
    cv2.putText(canvas, "Normalized field coordinates (0-1)", (margin, height-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
    return canvas