from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# green colorbar
cdict1 = {'red': ((0.0, 0.0, 0.0),
                  (0.0, 0.0, 0.0),
                  (1.0, 0.0, 0.0)),

          'green': ((0.0, 0.0, 0.0),
                    (0.15, 0.0, 0.0),
                    (1.0, 1.0, 1.0)),

          'blue': ((0.0, 0.0, 0.0),
                   (0.15, 0.0, 0.0),
                   (1.0, 0.5, 0.5))
          }
green = LinearSegmentedColormap('green', cdict1)

_jet_data ={'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                     (1, 0.5, 0.5)),
            'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                     (0.91,0,0), (1, 0, 0)),
            'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                     (1, 0, 0))}

cmap_data = {'jet': _jet_data}


def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)


jet_color = make_cmap('jet')

