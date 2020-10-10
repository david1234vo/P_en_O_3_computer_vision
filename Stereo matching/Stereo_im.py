import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


video_capture1 = cv2.VideoCapture(1)
video_capture2 = cv2.VideoCapture(2)

while True:
    ret, imgL = video_capture1.read()
    ret, imgR = video_capture2.read()

    stereo = cv2.StereoSGBM_create(numDisparities=16*1, blockSize=15,disp12MaxDiff = 2,
            uniquenessRatio = 5) # 16 15
    disparity = stereo.compute(imgL,imgR)
    cv2.imshow("imgR",imgR)
    cv2.imshow("imgL",imgL)

    disp = stereo.compute(imgL, imgR).astype(np.float32)

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    plt.imshow(disparity,'gray')
    plt.show(block=False)
    time.sleep(10)
    plt.close()
    # if cv2.waitKey("q"):
    #     cv2.destroyAllWindows()
    #     break
