import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import skew


def pol2car(rho, pi, xi, yi):
    x = rho * np.cos(pi) + xi
    y = rho * np.sin(pi) + yi
    return (x, y)

def car2pol(xi, yi):
    rho = np.sqrt(xi**2 + yi**2)
    phi = np.arctan2(yi, xi)
    return (rho, phi)

def car2polnorm(xi, yi):
    rho = np.sqrt(xi**2 + yi**2)
    phi = np.arctan2(yi, xi)
    phi %= 2*np.pi
    phi /= 2*np.pi
    return (rho, phi)

def plot_isovist(isovists, show_axis=False, s=0.1, figsize=(5,5)):
    #transpose the matrix
    # isovists = np.transpose(isovists, (isovists.ndim-1, isovists.ndim-2))
    plt.switch_backend('agg')
    fig = plt.figure(figsize=figsize)
    points = []
    res = np.pi/90
    isovist = isovists
    for j, rho in enumerate(isovist):
        if rho < 1.0:
            pt = pol2car(rho, j*res, 0, 0)
            points.append(pt)
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    if not show_axis:
        ax.axis('off')
    ax.scatter(x, y, s, 'black')
    return fig

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def isovist_to_img(isovist, show_axis=False, s=0.1, figsize=(5,5)):
    points = []
    xy = (0, 0)
    res = np.pi/90
    isovist = isovist + 0.5
    for j, rho in enumerate(isovist):
            if rho <= 2.0:
                pt = pol2car(rho, j*res, xy[0], xy[1])
                points.append(pt)
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    fig = plt.figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    if not show_axis:
        ax.axis('off')
    ax.scatter(x, y, s, 'black')

    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    return image

def isovist_to_img_a(isovist, show_axis=False, s=0.1, figsize=(5,5)):
    points = []
    xy = (0, 0)
    res = np.pi/128
    isovist = isovist + 0.5
    for j, rho in enumerate(isovist):
            if rho <= 2.0:
                pt = pol2car(rho, j*res, xy[0], xy[1])
                points.append(pt)
    x = [i[0] for i in points]
    y = [i[1] for i in points]
    fig = plt.figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    if not show_axis:
        ax.axis('off')
    ax.scatter(x, y, s, 'black')

    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    return image

def isovist_to_cartesian(isovist, x, y, scale):
    points = []
    xy = (x, y)
    res = np.pi/90
    isovist = isovist * scale
    for j, rho in enumerate(isovist):
        if rho <= scale:
            pt = pol2car(rho, j*res, xy[0], xy[1])
            points.append(pt)
        else:
            pt = pol2car(scale, j*res, xy[0], xy[1])
            points.append(pt)
    points = np.stack(points)
    return(points)

def isovist_to_cartesian_a(isovist, x, y, scale):
    points = []
    xy = (x, y)
    res = np.pi/len(isovist)*2
    isovist = isovist * scale
    for j, rho in enumerate(isovist):
        pt = pol2car(rho, j*res, xy[0], xy[1])
        points.append(pt)
    points = np.stack(points)
    return(points)

def isovist_to_cartesian_b(isovist, x, y):
    points = []
    xy = (x, y)
    res = np.pi*2
    isovist = isovist
    for j, rho in isovist:
        pt = pol2car(rho, j*res, xy[0], xy[1])
        points.append(pt)
    points = np.stack(points)
    return(points)

def isovist_to_cartesian_segment(isovist, x, y, scale):
    points = []
    segment = []
    xy = (x, y)
    res = np.pi/90
    isovist = isovist * scale
    p_rho = isovist[-1]
    for j, rho in enumerate(isovist):
        delta = abs(p_rho-rho)
        if j == 0:
            first_rho = rho
        if rho < 0.98 * scale and delta < 0.05 * scale:
            pt = pol2car(rho, j*res, xy[0], xy[1])
            segment.append(pt)
        else:
            points.append(segment)
            segment = []
        p_rho = rho
    if first_rho < 1.0 * scale and abs(rho-first_rho)< 0.05 * scale :
        if len(points) > 0:
            segment.extend(points[0])
            points[0]=segment
        else:
            points.append(segment)
    else:
        points.append(segment)
    segments = []
    for i in range(len(points)):
        if len(points[i])>0:
            segment = np.stack(points[i])
            segments.append(segment)
    return(segments)

def isovist_to_cartesian_segment_a(isovist, x, y, scale, max=0.98, min = 0.1, d=0.1):
    points = []
    segment = []
    xy = (x, y)
    res = np.pi/len(isovist)*2
    isovist = isovist * scale
    p_rho = isovist[-1]
    for j, rho in enumerate(isovist):
        delta = abs(p_rho-rho)
        if j == 0:
            first_rho = rho
        if rho < max * scale and rho > min * scale and delta < d * scale:
            pt = pol2car(rho, j*res, xy[0], xy[1])
            segment.append(pt)
        else:
            points.append(segment)
            segment = []
        p_rho = rho
    if first_rho < max * scale and  first_rho > min * scale and abs(rho-first_rho)< d * scale :
        if len(points) > 0:
            segment.extend(points[0])
            points[0]=segment
        else:
            points.append(segment)
    else:
        points.append(segment)
    segments = []
    for i in range(len(points)):
        if len(points[i])>0:
            segment = np.stack(points[i])
            segments.append(segment)
    return(segments)


def isovist_to_cartesian_segment_b(isovist, x, y):
    points = []
    segment = []
    xy = (x, y)
    res = np.pi*2
    isovist = isovist
    p_rho = isovist[-1, 1]
    _i = 0
    for j, rho in isovist:
        delta = abs(p_rho-rho)
        if _i == 0:
            first_rho = rho
        if rho < 0.98 and delta < 0.025 :
            pt = pol2car(rho, j*res, xy[0], xy[1])
            segment.append(pt)
        else:
            points.append(segment)
            segment = []
        p_rho = rho
        _i += 1
    if first_rho < 0.98  and abs(rho-first_rho)< 0.025:
        if len(points) > 0:
            segment.extend(points[0])
            points[0]=segment
        else:
            points.append(segment)
    else:
        points.append(segment)
    segments = []
    for i in range(len(points)):
        if len(points[i])>0:
            segment = np.stack(points[i])
            segments.append(segment)
    return(segments)


# plotting an isovist and return the numpy image
def plot_isovist_numpy(k, text=None, figsize=(8,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=300)

    #plot isovist
    xy = isovist_to_cartesian_a(k, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#dddddd')
    p.set_edgecolor(None)
    ax.add_collection(p)

    # style
    ax.set_aspect('equal')
    lim = 1.2
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    if text != None:
        ax.set_title(text, size=5) # Title
        fig.tight_layout()

    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im



# plotting isovist and boundary from  and return the numpy image
def plot_isovist_boundary_numpy(isovist, boundary, figsize=(8,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=300)

    #plot isovist
    xy = isovist_to_cartesian_a(isovist, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#eeeeee')
    p.set_edgecolor(None)
    ax.add_collection(p)


    #plot assumed boundary
    edge_patches = []
    segments = isovist_to_cartesian_segment_a(boundary, 0, 0, 1.0)
    for segment in segments:
        polygon = Polygon(segment, False)
        edge_patches.append(polygon)
    p = PatchCollection(edge_patches)
    p.set_facecolor('none')
    p.set_edgecolor('#000000')
    p.set_linewidth(0.5)
    ax.add_collection(p)

    # style
    ax.set_aspect('equal')
    lim = 1.2
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im


# plotting two isovists (fill and edge) and return the numpy image
def plot_isovist_double_numpy(isovist1, isovist2, figsize=(8,8)):
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=300)

    #plot isovist1
    xy = isovist_to_cartesian_a(isovist1, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#dddddd')
    p.set_edgecolor(None)
    ax.add_collection(p)

    #plot isovist2 as boundary
    xy = isovist_to_cartesian_a(isovist2, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('none')
    p.set_edgecolor('#000000')
    p.set_linewidth(0.2)
    ax.add_collection(p)

    # style
    ax.set_aspect('equal')
    lim = 1.2
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im


# plotting two isovists (fill and edge) and return the numpy image
def plot_isovist_triple_numpy(isovists, locs, figsize=(8,8)):
    isovist1, isovist2, isovist3 = isovists
    loc1, loc2, loc3 = locs

    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=300)

    #plot isovist1
    xy = isovist_to_cartesian_a(isovist1, loc1[0], loc1[1], 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#ffdddd')
    p.set_edgecolor(None)
    ax.add_collection(p)

    #plot isovist2
    xy = isovist_to_cartesian_a(isovist2, loc2[0], loc2[1], 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#ddddff')
    p.set_edgecolor(None)
    ax.add_collection(p)

    #plot isovist3 as boundary
    xy = isovist_to_cartesian_a(isovist3, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('none')
    p.set_edgecolor('#000000')
    p.set_linewidth(0.2)
    ax.add_collection(p)

    ax.scatter([x[0] for x in locs], [x[1] for x in locs], c='k', s=8, marker='+')

    annotation = ['x1', 'x2', 'y']
    for i, anno in enumerate(annotation):
        ax.annotate(anno, (locs[i][0]+0.1, locs[i][1]), size=8)

    # style
    ax.set_aspect('equal')
    lim = 1.5
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im


# plotting the boundary segment of 2 channel isovist and return numpy image
def plot_boudaries_numpy(isovists, figsize=(8,8)):
    isovist1, isovist2 = isovists
    isovist3 = np.zeros(256)
    for i in range(256):
        a = isovist1[i]
        b = isovist2[i]
        if a > b and b > 0.1:
            isovist3[i] = b
        else:
            isovist3[i] = a

    segments1 = isovist_to_cartesian_segment_a(isovist1, 0, 0, 1.0)
    segments2 = isovist_to_cartesian_segment_a(isovist2, 0, 0, 1.0)

    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=300)


    #plot isovist
    xy = isovist_to_cartesian_a(isovist1, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#f5f5f5')
    p.set_edgecolor(None)
    ax.add_collection(p)

    #plot isovist
    xy = isovist_to_cartesian_a(isovist3, 0, 0, 1.0)
    polygon = Polygon(xy, True)
    p = PatchCollection([polygon])
    p.set_facecolor('#dedede')
    p.set_edgecolor(None)
    ax.add_collection(p)

    edge_patches = []
    for segment in segments2:
        polygon = Polygon(segment, False)
        edge_patches.append(polygon)
    p = PatchCollection(edge_patches)
    p.set_facecolor('none')
    p.set_edgecolor('#00ffff')
    p.set_linewidth(0.5)
    ax.add_collection(p)

    edge_patches = []
    for segment in segments1:
        polygon = Polygon(segment, False)
        edge_patches.append(polygon)
    p = PatchCollection(edge_patches)
    p.set_facecolor('none')
    p.set_edgecolor('#000000')
    p.set_linewidth(0.5)
    ax.add_collection(p)

    # style
    ax.set_aspect('equal')
    lim = 1.5
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im


# showing isovist sequence
def seq_show(locs, isovists, figsize=(8, 8)):
    # walk trough the sequence
    p_loc = np.array((0, 0))
    b_segments = []
    b_points = []
    isovists_pts = []
    res = np.pi/128
    p_loc = np.array([0,0])
    cartesian_locs = []
    for loc, isovist in zip(locs, isovists):
        rel_pos = np.asarray(pol2car(loc[0], loc[1]*2*np.pi, p_loc[0], p_loc[1]))
        for j, rho in enumerate(isovist):
            if rho < 0.98 :
                pt = pol2car(rho, j*res, rel_pos[0], rel_pos[1])
                b_points.append(pt)
        segments = isovist_to_cartesian_segment_a(isovist, rel_pos[0], rel_pos[1], 1.0)
        b_segments.extend(segments)
        isovists_pts.append(isovist_to_cartesian_a(isovist, rel_pos[0], rel_pos[1], 1.0))
        cartesian_locs.append(rel_pos)
        p_loc = rel_pos

    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=96)

    
    # isovists
    isovist_poly = []
    for isovist_pts in isovists_pts:
        isovist_poly.append(Polygon(isovist_pts, True))
    r = PatchCollection(isovist_poly)
    r.set_facecolor('#000000')
    r.set_edgecolor(None)
    r.set_alpha(0.02)
    ax.add_collection(r)
    

    # isovist path
    q = PatchCollection([Polygon(cartesian_locs, False)])
    q.set_facecolor('none')
    q.set_edgecolor('#cccccc')
    q.set_linewidth(1.0)
    q.set_linestyle('dashed')
    ax.add_collection(q)
    ax.scatter([x[0] for x in cartesian_locs], [x[1] for x in cartesian_locs], s = 6.0, c='red')

    # boundaries
    edge_patches = []
    for segment in b_segments:
        polygon = Polygon(segment, False)
        edge_patches.append(polygon)
    p = PatchCollection(edge_patches)
    p.set_facecolor('none')
    p.set_edgecolor('#000000')
    p.set_linewidth(1.0)
    ax.add_collection(p)
    ax.scatter([x[0] for x in b_points], [x[1] for x in b_points], s = 0.05, c='k')


    # style
    ax.set_aspect('equal')
    lim = 1.5
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return fig


# plotting isovist sequence
def plot_isovist_sequence(locs, isovists, figsize=(8,8)):
    fig = seq_show(locs, isovists, figsize=figsize)

    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im


def index_to_loc_grid(idx, d):
    if idx == 0:
        return np.array((0., 0.), dtype=np.float32)
    elif idx == 1:
        return np.array((d, 0.), dtype=np.float32)
    elif idx == 2:
        return np.array((d, d), dtype=np.float32)
    elif idx == 3:
        return np.array((0., d), dtype=np.float32)
    elif idx == 4:
        return np.array((-d, d), dtype=np.float32)
    elif idx == 5:
        return np.array((-d, 0.), dtype=np.float32)
    elif idx == 6:
        return np.array((-d, -d), dtype=np.float32)
    elif idx == 7:
        return np.array((0., -d), dtype=np.float32)
    elif idx == 8:
        return np.array((d, -d), dtype=np.float32)
    else:
        raise NameError('Direction unknown')


# showing isovist sequence grid
def seq_show_grid(locs, isovists, d=0.2, figsize=(8, 8), center=False, lim=1.5, alpha=0.02, rad=0.9, b_width=1.0, calculate_lim=False):
    # walk trough the sequence
    p_loc = np.array((0, 0))
    b_segments = []
    b_points = []
    isovists_pts = []
    res = np.pi/128
    cartesian_locs = []
    for loc, isovist in zip(locs, isovists):
        rel_pos = index_to_loc_grid(loc, d) + p_loc
        for j, rho in enumerate(isovist):
            if rho < rad :
                pt = pol2car(rho, j*res, rel_pos[0], rel_pos[1])
                b_points.append(pt)
        segments = isovist_to_cartesian_segment_a(isovist, rel_pos[0], rel_pos[1], 1.0)
        b_segments.extend(segments)
        isovists_pts.append(isovist_to_cartesian_a(isovist, rel_pos[0], rel_pos[1], 1.0))
        cartesian_locs.append(rel_pos)
        p_loc = rel_pos

    if len(b_points) > 0:
        b_points = np.stack(b_points)
    else:
        b_points =[]
    isovists_pts = np.stack(isovists_pts)
    # b_segments = np.stack(b_segments)
    cartesian_locs = np.stack(cartesian_locs)

    # set graphic properties
    isovist_path_width = 0.1
    isovist_path_pt1 = 6.0
    isovist_path_pt2 = 10.0
    isovist_boundary_pt = 0.05

    if center == True:

        bbox = get_bbox(b_points)
        center_pt = get_center_pts(bbox, np_array=True)
        b_points = [ pt - center_pt for pt in b_points]
        isovists_pts = [ pt - center_pt for pt in isovists_pts]
        b_segments =  [ pt - center_pt for pt in b_segments]
        cartesian_locs = [ pt - center_pt for pt in cartesian_locs]

    # resize image
    if calculate_lim == True:
        if bbox is not None:
            max = np.max(np.abs(bbox))
        else:
            max = 2.0
        if max > 2.0:
            lim = ((max // 0.5) + 1) * 0.5
            isovist_path_width *= 2.0/lim
            isovist_path_pt1 *= 2.0/lim
            isovist_path_pt2 *= 2.0/lim
            isovist_boundary_pt *= 2.0/lim


    fig, ax = plt.subplots(1,1, figsize=figsize, dpi=96)

    

    # isovists
    isovist_poly = []
    for isovist_pts in isovists_pts:
        isovist_poly.append(Polygon(isovist_pts, True))
    r = PatchCollection(isovist_poly)
    r.set_facecolor('#00aabb')
    r.set_edgecolor(None)
    r.set_alpha(alpha)
    ax.add_collection(r)

    

    # isovist path
    q = PatchCollection([Polygon(cartesian_locs, False)])
    q.set_facecolor('none')
    q.set_edgecolor('red')
    q.set_linewidth(isovist_path_width)
    # q.set_linestyle('dashed')
    ax.add_collection(q)
    
    # start_pt
    ax.scatter([x[0] for x in cartesian_locs[:1]], [x[1] for x in cartesian_locs[:1]], s = isovist_path_pt1, c='k', marker='s')

    # sequence
    ax.scatter([x[0] for x in cartesian_locs[1:-1]], [x[1] for x in cartesian_locs[1:-1]], s = isovist_path_pt1, c='red')

    # end pt
    ax.scatter([x[0] for x in cartesian_locs[-1:]], [x[1] for x in cartesian_locs[-1:]], s = isovist_path_pt2, c='k', marker='x')

    # boundaries
    edge_patches = []
    for segment in b_segments:
        if len(segment) > 5:
            polygon = Polygon(segment, False)
            edge_patches.append(polygon)
    p = PatchCollection(edge_patches)
    p.set_facecolor('none')
    p.set_edgecolor('#000000')
    p.set_linewidth(b_width)
    ax.scatter([x[0] for x in b_points], [x[1] for x in b_points], s = isovist_boundary_pt, c='#000000',)
    # ax.add_collection(p)


    # style
    ax.set_aspect('equal')
    lim = lim
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    return fig

# plotting isovist sequence grid
def plot_isovist_sequence_grid(locs, isovists, figsize=(8,8), center=False, lim=1.5, alpha=0.02, rad=0.9, b_width=1.0, calculate_lim=False):
    fig = seq_show_grid(locs, isovists, figsize=figsize, center=center, lim=lim, alpha=alpha, rad=rad, b_width=b_width, calculate_lim=calculate_lim)
    # for plot with torchvision util
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))
    plt.close()
    return im

def get_bbox(pts):
    if len(pts) > 0:
        if type(pts) is list:
            pts = np.stack(pts)
        bbox = np.min(pts[:, 0]), np.max(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 1])
        return bbox
    else:
        return None

def get_center_pts(bbox, np_array = False):
    if bbox is not None:
        center = 0.5*(bbox[0] + bbox[1]),  0.5*(bbox[2] + bbox[3])
        if np_array:
            center = np.asarray(center)
    else:
        center = np.asarray([0,0])
    return center



# set of isovist measures
def isovist_area(isovist):
    res = len(isovist)
    area = np.pi*np.square(isovist)/res
    return np.sum(area)

def isovist_perimeter(isovist):
    res = len(isovist)
    rads = np.array([2*i*np.pi/res for i in range(res)])
    rads_shift = np.roll(rads, 1)
    isovist_shift = np.roll(isovist, 1)
    delta_rho = np.abs(isovist-isovist_shift)
    min_perim = np.minimum(isovist, isovist_shift)*np.pi*2/res
    angles = np.arctan2(delta_rho, min_perim)
    x1, y1 = np.cos(rads)*isovist, np.sin(rads)*isovist
    x2, y2 = np.cos(rads_shift)*isovist_shift, np.sin(rads_shift)*isovist_shift
    xy1 = np.stack([x1, y1], axis=1)
    xy2 = np.stack([x2, y2], axis=1)
    segment_length = np.linalg.norm((xy1-xy2), axis=1)
    real_perimeter = 0
    occlusivity = 0
    horizon = 0
    total_perimeter = 0

    for i, angle in enumerate(angles):
        if isovist[i] <= 0.98 and isovist_shift[i] <= 0.98 and angle/np.pi*180 < 85:
            real_perimeter += segment_length[i]
        if angle/np.pi*180 >= 85 or ((isovist[i] > 0.98) ^ (isovist_shift[i] >0.98)):
            occlusivity += segment_length[i]
        if isovist[i] > 0.98  and isovist_shift[i] > 0.98:
            seg = np.array([(x1[i], y1[i]), (x2[i], y2[i])])
            horizon += segment_length[i]
    total_perimeter = np.sum(segment_length)

    return real_perimeter, occlusivity, horizon, total_perimeter


def isovist_perimeter2(isovist):
    res = len(isovist)
    rads = np.array([2*i*np.pi/res for i in range(res)])
    rads_shift = np.roll(rads, 1)
    isovist_shift = np.roll(isovist, 1)
    delta_rho = np.abs(isovist-isovist_shift)
    sum_rho = isovist+isovist_shift
    angles = delta_rho/sum_rho
    x1, y1 = np.cos(rads)*isovist, np.sin(rads)*isovist
    x2, y2 = np.cos(rads_shift)*isovist_shift, np.sin(rads_shift)*isovist_shift
    xy1 = np.stack([x1, y1], axis=1)
    xy2 = np.stack([x2, y2], axis=1)
    segment_length = np.linalg.norm((xy1-xy2), axis=1)
    real_perimeter = 0
    occlusivity = 0
    horizon = 0
    total_perimeter = 0

    for i, angle in enumerate(angles):
        if isovist[i] <= 1.0 and isovist_shift[i] <= 1.0 and angle < 0.15:
            real_perimeter += segment_length[i]
        if angle >=0.15:
            occlusivity += segment_length[i]
        if isovist[i] == 1.0  and isovist_shift[i] == 1.0:
            seg = np.array([(x1[i], y1[i]), (x2[i], y2[i])])
            horizon += segment_length[i]
    total_perimeter = np.sum(segment_length)

    return real_perimeter, occlusivity, horizon, total_perimeter


def isovist_variance(isovist):
    return np.var(isovist)

def isovist_skewness(isovist):
    return skew(isovist)

def isovist_circularity(isovist):
    area = isovist_area(isovist)
    _, _, _, total_perimeter = isovist_perimeter(isovist)
    return total_perimeter**2 / (4*np.pi*area)

def isovist_drift(isovist):
    res = len(isovist)
    rads = np.array([2*i*np.pi/res for i in range(res)])
    x, y = np.cos(rads)*isovist, np.sin(rads)*isovist
    isovist_polygon = np.stack([x,y], axis=1)
    isovist_polygon_shift = np.roll(isovist_polygon, -1, axis=0)
    # Compute signed area of each triangle
    signed_areas = 0.5 * np.cross(isovist_polygon, isovist_polygon_shift)
    # Compute centroid of each triangle
    centroids = (isovist_polygon + isovist_polygon_shift) / 3.0
    # Get average of those centroids, weighted by the signed areas.
    centroid = np.average(centroids, axis=0, weights=signed_areas)
    return np.linalg.norm(centroid)