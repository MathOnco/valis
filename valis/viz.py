"""Various functions used to visualize registration results

"""
import colour
import matplotlib.pyplot as plt
from skimage import feature, draw, color, exposure, transform
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial import distance
import numpy as np
import numba as nb
from . import warp_tools

# JzAzBz #
DXDY_CSPACE = "JzAzBz"
DXDY_CRANGE = (0, 0.025)
DXDY_LRANGE = (0.004, 0.015)

# CAM16-UCS #
# DXDY_CSPACE = "CAM16UCS"
# DXDY_CRANGE = (0, 0.5)
# DXDY_LRANGE = (0.5, 0.9)


def draw_features(kp_xy, image, n_features=500):
    """Draw keypoints on a image

    """

    image = exposure.rescale_intensity(image, out_range=(0, 255))
    if image.ndim == 2:
        feature_img = color.grey2rgb(image)
    else:
        feature_img = image

    rad = int(np.mean(feature_img.shape) / 100)

    n_kp = len(kp_xy)
    if n_kp < n_features:
        n_features = n_kp

    for c, r in kp_xy[0:n_features].astype(np.int):
        circ_r, circ_c = draw.circle_perimeter(r, c, rad, shape=image.shape[0:2])
        feature_img[circ_r, circ_c] = np.random.randint(0, 256, 3)

    return feature_img


def draw_matches(src_img, kp1_xy, dst_img, kp2_xy, alignment='horizontal'):
    """
    Draw all matches between src_img and dst_img, using scikit-image. Assumes they have already been filtered
    Parameters
    ----------
        src_img : ndarray
            Image from which kp1_xy were detected

        kp1_xy : (N, 2) array
            Image 1s keypoint positions, in xy coordinates,  for each of the N descriptors in desc1

        dst_img : ndarray
            Image from which kp2_xy were detected

        kp2_xy : (M, 2) array
            Image 1s keypoint positions, in xy coordinates,  for each of the M descriptors in desc2
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    plt.gray()
    match_idx = np.arange(0, len(kp1_xy))
    matches = np.dstack([match_idx, match_idx])[0]

    feature.plot_matches(ax, src_img, dst_img, kp1_xy[:, ::-1], kp2_xy[:, ::-1], matches, alignment=alignment)
    plt.title(" ".join([str(len(kp1_xy)), "matches"]))
    ax.axis('off')
    plt.tight_layout()
    # plt.close()


def draw_clusterd_D(D, optimal_Z):
    """Draw clustered distance matrix with dendrograms along the axes

    """

    fig = plt.figure()
    axdendro = fig.add_axes([0.013, 0.05, 0.1, 0.798])

    Z = dendrogram(optimal_Z, orientation='left', link_color_func=lambda k: "black")
    axdendro.set_xticks([])
    axdendro.set_yticks([])
    axdendro.axis('off')
    axdendro.invert_yaxis()

    axdendro_top = fig.add_axes([0.115, 0.85, 0.6, 0.14])
    Z_top = dendrogram(optimal_Z, orientation='top', link_color_func=lambda k: "black")
    axdendro_top.set_xticks([])
    axdendro_top.set_yticks([])
    axdendro_top.axis('off')

    # axmatrix = fig.add_axes([0.2, 0.1, 0.6, 0.8])
    axmatrix = fig.add_axes([0.115, 0.05, 0.6, 0.798])
    im = axmatrix.matshow(D, aspect='auto', cmap="plasma_r")
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # axcolor = fig.add_axes([0.82, 0.05, 0.02, 0.8])
    axcolor = fig.add_axes([0.75, 0.05, 0.03, 0.798])
    plt.colorbar(im, cax=axcolor)


# Non-rigid visualization #
@nb.njit(nb.typeof((np.array([1]), np.array([1])))(nb.typeof((1, 1)), nb.typeof(10), nb.typeof(1)))
def get_grid(shape, grid_spacing, thickness=1):
    """
    Get points for a grid. Can be used to view deformation field

    Parameters
    ----------
    shape : (int, int)
        dimensions of image upon which the grid will be drawn

    grid_spacing : int
        Space between grid points

    thickness : int, optional
        line thickness

    Returns
    -------
    grid_rows, grid_cols : 2 ndarray
        2, 1D arrays, which each element corresponding to a point in the grid
    """

    all_rows =[]
    all_cols = []
    # thickness = 2
    row_add_idx = 0
    for k in range(thickness):
        # for i in np.arange(thickness//2, shape[0] + grid_spacing, grid_spacing):
        for i in np.arange(grid_spacing - thickness, shape[0] + thickness, grid_spacing):
            for j in np.arange(0, shape[1]):
                if k%2 == 0:
                    r = i + row_add_idx
                elif k%2 != 0:
                    r = i - row_add_idx

                if r >= 0 and r < shape[0]:
                    all_rows.append(r)
                    all_cols.append(j)

        if k % 2 == 0:
            row_add_idx += 1

    col_add_idx = 0
    for k in range(thickness):
        # for j in np.arange(thickness//2, shape[1] + grid_spacing, grid_spacing):
        for j in np.arange(grid_spacing - thickness, shape[1], grid_spacing):
            for i in np.arange(0, shape[0]):
                if k%2 == 0:
                    c = j + col_add_idx
                elif k%2 != 0:
                    c = j - col_add_idx

                if c >= 0 and c < shape[1]:
                    all_rows.append(i)
                    all_cols.append(c)

        if k % 2 == 0:
            col_add_idx += 1

    return np.array(all_rows), np.array(all_cols)


def jzazbz_cmap(luminosity=0.012, colorfulness=0.02, max_h=260):
    """
    Get colormap based on JzAzBz colorspace, which has good hue linearity.
    Already preceptually uniform.

    Parameters
    ----------
    luminosity :  float, optional

    colorfulness : float, optional

    max_h : int, optional

    """

    h = np.deg2rad(np.arange(0, 360))
    a = colorfulness * np.cos(h)
    b = colorfulness * np.sin(h)
    j = np.repeat(luminosity, len(h))

    jzazbz = np.dstack([j, a, b])
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jzazbz, 'JzAzBz', 'sRGB')

    rgb = np.clip(rgb, 0, 1)[0]
    max_h = 260
    if max_h != 360:
        rgb = rgb[0:max_h]

    return rgb

def cam16ucs_cmap(luminosity=0.8, colorfulness=0.5, max_h=300):
    """
    Get colormap based on CAM16-UCS colorspace.

    Parameters
    ----------
    luminosity :  float, optional

    colorfulness : float, optional

    max_h : int, optional

    """

    h = np.deg2rad(np.arange(0, 360))
    a = colorfulness * np.cos(h)
    b = colorfulness * np.sin(h)
    j = np.repeat(luminosity, len(h))

    eps = np.finfo("float").eps
    cam = np.dstack([j, a+eps, b+eps])
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(cam, 'CAM16UCS', 'sRGB')

    rgb = np.clip(rgb, 0, 1)[0]
    if max_h != 360:
        rgb = rgb[0:max_h]

    return rgb



def rgb_triangle_cmap():

    total_n = 360
    n = total_n//3
    max_v = 0.9
    min_v = 1 - max_v

    tri_edges_x = np.hstack([np.linspace(min_v, max_v, n*2), np.linspace(max_v - 1/n, min_v + 1/n, n)])
    tri_edges_y = np.hstack([np.linspace(min_v, max_v, n), np.linspace(max_v - 1/n, min_v, n), np.repeat(min_v, n)])

    tri_edges_xy = np.dstack([tri_edges_x, tri_edges_y])[0]

    T = np.array([[-1., -0.5],
                  [0., 1.]])

    bary_xy = np.array([np.linalg.inv(T) @ (tri_edges_xy[i] - np.array([ 1.,  0.])) for i in range(len(tri_edges_xy))])
    bary_z = 1 - np.sum(bary_xy, axis=1)
    rgb = np.array([bary_xy[:, 0], bary_xy[:, 1], bary_z]).T

    return rgb


def turbo_cmap():
    """Turbo colormap
    https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
    """
    # The look-up table contains 256 entries. Each entry is a floating point sRGB triplet.
    # To use it with matplotlib, pass cmap=ListedColormap(turbo_colormap_data) as an arg to imshow() (don't forget "from matplotlib.colors import ListedColormap").
    # If you have a typical 8-bit greyscale image, you can use the 8-bit value to index into this LUT directly.
    # The floating point color values can be converted to 8-bit sRGB via multiplying by 255 and casting/flooring to an integer. Saturation should not be required for IEEE-754 compliant arithmetic.
    # If you have a floating point value in the range [0,1], you can use interpolate() to linearly interpolate between the entries.
    # If you have 16-bit or 32-bit integer values, convert them to floating point values on the [0,1] range and then use interpolate(). Doing the interpolation in floating point will reduce banding.
    # If some of your values may lie outside the [0,1] range, use interpolate_or_clip() to highlight them.

    turbo_colormap_data = np.array([[0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024],
                           [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314],
                           [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096],
                           [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373],
                           [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142],
                           [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406],
                           [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162],
                           [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412],
                           [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156],
                           [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393],
                           [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123],
                           [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347],
                           [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064],
                           [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275],
                           [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930],
                           [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773],
                           [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835],
                           [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202],
                           [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959],
                           [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190],
                           [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981],
                           [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416],
                           [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580],
                           [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559],
                           [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437],
                           [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299],
                           [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230],
                           [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316],
                           [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599],
                           [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627],
                           [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308],
                           [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713],
                           [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914],
                           [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981],
                           [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987],
                           [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002],
                           [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098],
                           [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345],
                           [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816],
                           [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581],
                           [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712],
                           [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280],
                           [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [0.64362, 0.98999, 0.23356],
                           [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960],
                           [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032],
                           [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504],
                           [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311],
                           [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386],
                           [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663],
                           [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074],
                           [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555],
                           [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038],
                           [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456],
                           [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744],
                           [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836],
                           [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663],
                           [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161],
                           [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358],
                           [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348],
                           [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165],
                           [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842],
                           [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412],
                           [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910],
                           [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367],
                           [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817],
                           [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294],
                           [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831],
                           [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461],
                           [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218],
                           [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134],
                           [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243],
                           [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521],
                           [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875],
                           [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305],
                           [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809],
                           [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387],
                           [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041],
                           [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769],
                           [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571],
                           [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449],
                           [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401],
                           [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427],
                           [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529],
                           [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705],
                           [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955],
                           [0.47960, 0.01583, 0.01055]])

    return turbo_colormap_data


def get_n_colors(rgb, n):
    """
    Pick n most different colors in rgb. Differences based of rgb values in the CAM16UCS colorspace
    Based on https://larssonjohan.com/post/2016-10-30-farthest-points/
    """
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < rgb.max() <= 255 and np.issubdtype(rgb.dtype, np.integer):
            cam = colour.convert(rgb/255, 'sRGB', 'CAM16UCS')
        else:
            cam = colour.convert(rgb, 'sRGB', 'CAM16UCS')

    # sq_D = distance.cdist(cam, cam, metric=colour.difference.delta_E_CAM16UCS)
    sq_D = distance.cdist(cam, cam)
    max_D = sq_D.max()
    most_dif_2Didx = np.where(sq_D == max_D) ### 2 most different colors
    most_dif_img1 = most_dif_2Didx[0][0]
    most_dif_img2 = most_dif_2Didx[1][0]
    rgb_idx = [most_dif_img1, most_dif_img2]

    possible_idx = list(range(sq_D.shape[0]))
    possible_idx.remove(most_dif_img1)
    possible_idx.remove(most_dif_img2)

    for new_color in range(2, n):
        max_d_idx = np.argmax([np.min(sq_D[i, rgb_idx]) for i in possible_idx])
        new_rgb_idx = possible_idx[max_d_idx]
        rgb_idx.append(new_rgb_idx)
        possible_idx.remove(new_rgb_idx)

    return rgb[rgb_idx]


@nb.njit()
def get_AB(img, a, b):

    eps = 1.0000000000000001e-15
    A = np.zeros(img.shape[0:2])
    B = np.zeros(img.shape[0:2])
    if img.ndim > 2:
        sum_img = np.sum(img, axis=2) + eps  # Avoid division by 0
        for i in range(img.shape[2]):
            chanel_weight = img[..., i]/sum_img
            A += a[i] * chanel_weight
            B += b[i] * chanel_weight
    else:
        sum_img = img/(img.max())
        A = a * sum_img
        B = b * sum_img

    return A, B


def color_multichannel(multichannel_img, marker_colors, rescale_channels=False, normalize_by="channel", cspace="CAM16UCS"):
    """Color a multichannel image to view as RGB

    Parameters
    ----------
    multichannel_img : ndarray
        Image to color

    marker_colors : ndarray
        RGB colors for each channel. These RGB values are between 0-1, not 0-255

    rescale_channels : bool
        If True, then each channel will be scaled between 0 and 1 before building the composite RGB image. This will
        allow markers to 'pop' in areas where they are expressed in isolation, but can also make it appear more marker
        is expressed than there really is.

    normalize_by : str, optionaal
        "image" will produce an image where all values are scaled between 0 and the highest intensity in the composite image.
        This will produce an image where one can see the expression of each marker relative to the others, making it easier to
        compare marker expression levels.

        "channel" will first scale the intensity of each channel, and then blend all of the channels together. This will allow
        one to see the relative expression of each marker, but won't allow one to directly compare the expression of markers
        across channels.

    cspace : str
        Colorspace in which `marker_colors` will be blended.
        See the "color-science" package for details.

    Returns
    -------
    rgb : ndarray
        An RGB version of `multichannel_img`

    """

    if rescale_channels:
        multichannel_img = np.dstack([exposure.rescale_intensity(multichannel_img[..., i].astype(float), in_range="image", out_range=(0, 1)) for i in range(multichannel_img.shape[2])])

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < marker_colors.max() <= 255 and np.issubdtype(marker_colors.dtype, np.integer):
            jab_colors = colour.convert(marker_colors/255, 'sRGB', cspace)
        else:
            jab_colors = colour.convert(marker_colors, 'sRGB', cspace)


    A, B = get_AB(multichannel_img, jab_colors[..., 1], jab_colors[..., 2])

    if multichannel_img.ndim > 2:
        if normalize_by == "channel":
            J = np.max(multichannel_img, axis=2)
        if normalize_by == "image":
            J = np.sum(multichannel_img, axis=2)
    else:
        J = multichannel_img.copy()

    J = J/J.max()
    if cspace == "JzAzBz":
        J*= 0.01 #0.025

    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(np.dstack([J, A+eps, B+eps]), cspace, 'sRGB')

    rgb = (255*np.clip(rgb, 0, 1)).astype(np.uint8)

    return rgb


def color_dxdy(dx, dy, c_range=DXDY_CRANGE, l_range=DXDY_LRANGE, cspace=DXDY_CSPACE):
    """
   Color displacement, where larger displacements are more colorful,
   and, if scale_l=True,  brighter.

    Parameters
    ----------
    dx: array
        1D Array containing the displacement in the X (column) direction

    dy: array
        1D Array containing the displacement in the Y (row) direction

    c_range: (float, float)
        Minimum and maximum colorfulness in JzAzBz colorspace

    l_range: (float, float)
        Minimum and maximum luminosity in JzAzBz colorspace

    scale_l: boolean
        Scale the luminosity based on magnitude of displacement

    Returns
    -------
    displacement_rgb : array
        RGB (0, 255) color for each displacement, with the same shape as dx and dy

    """

    initial_shape = dx.shape

    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    if np.all(dx==0) and np.all(dy==0):
        # No displacements. Return grey image
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            bg_rgb = colour.convert(np.dstack([l_range[0], 0, 0]), cspace, 'sRGB')*255

        displacement_rgb = np.full((*initial_shape, 3), bg_rgb).astype(np.uint8)
        return displacement_rgb

    eps = np.finfo("float").eps
    magnitude = np.sqrt(dx ** 2 + dy ** 2 + eps)
    C = exposure.rescale_intensity(magnitude, in_range=(0, magnitude.max()), out_range=tuple(c_range))
    H = np.arctan2(dy.T, dx.T)
    A, B = C * np.cos(H), C * np.sin(H)
    J = exposure.rescale_intensity(magnitude, in_range=(0, magnitude.max()), out_range=tuple(l_range))

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(np.dstack([J, A+eps, B+eps]), cspace, 'sRGB')

    displacement_rgb = (255*np.clip(rgb, 0, 1)).astype(np.uint8).reshape((*initial_shape, 3))

    return displacement_rgb


def displacement_legend():

    X = np.linspace(-1, 1, 100)
    Y = np.linspace(-1, 1, 100)

    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    C = np.sin(R)

    C = exposure.rescale_intensity(C, out_range=(0, 1))

    grad = np.linspace(-1, 1, X.shape[0])
    grad = np.resize(grad, X.shape)
    dx = grad*C
    dy = grad.T * C

    displacement_legend = color_dxdy(dx, dy, DXDY_CRANGE, DXDY_LRANGE, cspace=DXDY_CSPACE)

    return displacement_legend


def draw_displacement_legend():
    leg = displacement_legend()
    fig, ax = plt.subplots()
    plt.locator_params(nbins=10)
    ax.imshow(leg)
    ax.set_xticklabels(["", "--", "", "", "", "", "0", "", "", "", "+"])
    ax.set_yticklabels(["", "+", "", "", "", "", "0", "", "", "", "--"])
    ax.set_xlabel('dx')
    ax.set_ylabel('dy')


def color_displacement_grid(bk_dx, bk_dy, c_range=DXDY_CRANGE, l_range=DXDY_LRANGE, thickness=None, grid_spacing_ratio=0.02, cspace=DXDY_CSPACE):
    """Color a displacement grid
    """

    grid_spacing = np.max(np.array(bk_dx.shape)*grid_spacing_ratio).astype(int)
    min_dim = np.min(bk_dx.shape)

    if thickness is None:
        thickness = int(np.ceil((grid_spacing/min_dim)*15))

    if thickness < 1:
        thickness = 1

    grid_r, grid_c = get_grid(bk_dx.shape, grid_spacing, thickness)
    grid_colors = color_dxdy(bk_dx[grid_r, grid_c], bk_dy[grid_r, grid_c],  c_range=c_range, l_range=l_range, cspace=cspace)

    # Warp image of grid
    grid_img = np.zeros((*bk_dx.shape, 3))
    grid_img[grid_r, grid_c] = grid_colors

    img_r, img_c = np.indices(bk_dx.shape)
    img_warp_r = img_r + bk_dy[img_r, img_c]
    img_warp_c = img_c + bk_dx[img_r, img_c]

    img_warp_r = np.clip(img_warp_r, 0, bk_dx.shape[0] - 1)
    img_warp_c = np.clip(img_warp_c, 0, bk_dx.shape[1] - 1)

    warped_hcl = [None] * 3
    for i in range(3):
        warped_hcl[i] = transform.warp(grid_img[..., i], np.array([img_warp_r, img_warp_c]))

    grid_img = np.dstack(warped_hcl).astype(np.uint8)


    return grid_img


def trapez(y,y0,w):
    return np.clip(np.minimum(y+1+w/2-y0, -y+1+w/2+y0),0,1)

def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1-c0) < abs(r1-r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax)

    # The following is now always < 1 in abs
    slope = (r1-r0) / (c1-c0)

    # Adjust weight by the slope
    w *= np.sqrt(1+np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1+1, dtype=float)
    y = x * slope + (c1*r0-c0*r1) / (c1-c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w/2)
    yy = (np.floor(y).reshape(-1,1) + np.arange(-thickness-1,thickness+2).reshape(1,-1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1,1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def draw_trimesh(shape_rc, tri_verts, tri_faces, thickness=2):
    """Draw a triangular mesh
    """

    def draw_line(tri_img, pt1_xy, pt2_xy):
        r, c, v = weighted_line(*pt1_xy[::-1], *pt2_xy[::-1], thickness)
        r = np.clip(r, 0, tri_img.shape[0]-1)
        c = np.clip(c, 0, tri_img.shape[1]-1)
        tri_img[r, c] = v

    tri_img = np.zeros(shape_rc)
    for face in tri_faces:
        l1_xy = tri_verts[face[0]]
        l2_xy = tri_verts[face[1]]
        l3_xy = tri_verts[face[2]]

        if not np.any(np.isnan(l1_xy)) and not np.any(np.isnan(l2_xy)):
            draw_line(tri_img, l1_xy, l2_xy)

        if not np.any(np.isnan(l2_xy)) and not np.any(np.isnan(l3_xy)):
            draw_line(tri_img, l2_xy, l3_xy)

        if not np.any(np.isnan(l3_xy)) and not np.any(np.isnan(l1_xy)):
            draw_line(tri_img, l3_xy, l1_xy)

    return tri_img


def color_displacement_tri_grid(bk_dx, bk_dy, n_grid_pts=25, c_range=DXDY_CRANGE, l_range=DXDY_LRANGE,  thickness=None, cspace=DXDY_CSPACE):
    """View how a displacement warps a triangular mesh.
    """

    shape = np.array(bk_dx.shape)
    grid_spacing = int(np.min(np.round(shape/n_grid_pts)))

    new_r = shape[0] - shape[0] % grid_spacing + grid_spacing
    sample_y = np.arange(0, new_r + grid_spacing, grid_spacing)

    new_c = shape[1] - shape[1] % grid_spacing + grid_spacing
    sample_x = np.arange(0, new_c + grid_spacing, grid_spacing)

    padded_shape = np.array([new_r+1, new_c+1])

    padding_T = warp_tools.get_padding_matrix(shape, padded_shape)

    padded_dx = transform.warp(bk_dx, padding_T, output_shape=padded_shape, preserve_range=True)
    padded_dy = transform.warp(bk_dy, padding_T, output_shape=padded_shape, preserve_range=True)


    min_dim = np.min(padded_dy.shape)
    if thickness is None:
        thickness = int(np.ceil((grid_spacing/min_dim)*15))
    if thickness < 1:
        thickness = 1


    tri_verts, tri_faces = warp_tools.get_triangular_mesh(sample_x, sample_y)
    warped_xy = warp_tools.warp_xy(tri_verts, bk_dxdy=[padded_dx, padded_dy])

    inv_T = np.linalg.inv(padding_T)
    trimesh_img = draw_trimesh(padded_shape, warped_xy, tri_faces, thickness=thickness)
    trimesh_img = transform.warp(trimesh_img, inv_T, output_shape=shape, preserve_range=True)
    colored_displacement = color_dxdy(bk_dx, bk_dy, c_range=c_range, l_range=l_range, cspace=cspace)
    colored_trimesh = trimesh_img[..., np.newaxis] * colored_displacement

    return colored_trimesh.astype(np.uint8)

