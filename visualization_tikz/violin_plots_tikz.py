import numpy as np
from scipy.stats import gaussian_kde

def violin_for_tikz(data, fillcolor='blue!40', plot_prop='smooth', center=0.0, abs_data=False, normalize=None,
                    quantiles=None, overshoot=True, draw_min_max=True, draw_mean=True, draw_median=True,
                    draw_quantiles=False, **kwargs):
    
    '''
    Draw a Violin plot using pfgplots/TikZ
    
    input:
        data: np.array() containing the data
        fillcolor: defines the fillcolor of the plot
        plot_prop: controls the properties of the smooth plotting that is used, don't need to add 'cycle'!
        center: defines the x center of the plot
        abs_data: Controls if only asolute value of data is considered. This might cause problems if we overshoot, then the kde has values in the negatives wich makes problems when scaling the area
        normalize: if None: no normalization
                   if 'area': the area under one half of the violin is normalized to normalize_fac (uses trapezoidal rul)
                   if 'max': the maximum value is normalized to normalize_fac
        quantiles: None or List, contains the quantiles [0,1] to be drawn
        overshoot: Controls if the plot should be extended above the min and max value, controlled by kwargs overshoot_top and overshoot_bottom
        draw_min_max: if True: min and max value is drawn using horizontal lines, vertical line is added too
        draw_mean: if True: draw mean value using a horizontal line
        draw_median: if True: draw median using a horizonal line
        draw_quantiles: if True: draw quantiles using horizontal lines
        
        kwargs:
            points: controls number of points being used (100)
            bw: defines bw_method for kde, see scipy.stats.gaussian_kde
            cut_bottom: controls, how much the graph should overshoot at the bottom (5)
                              overshoot = (std of the data) * (kde bandwidth) * (overshoot bottom)
            cut_top: same as overshoot_bottom, but for the top (maximum) value (5)                 
            normalize_fac: see normalize (1.0)
            max_width: defines width of maximum line (value of kde)
            max_prop: defines properties for the max line ('black')
            min_width: defines width of minimum line (value of kde)
            min_prop: defines properties for the min line ('thick, black')
            mvert_prop: defines properties for the vertical line ('thick, black')
            mean_width: defines width of mean line (value of kde)
            mean_prop: defines properties for the mean line ('thick, black')
            median_width: defines width of median line (value of kde)
            median_prop: defines properties for the median line ('thick, black, dashed')
            flat_top_bot: control if bottom and top are flat ('False')
            full_output: if True, prints additional info for legend and pic
            quant_width: defines width of quantile lines (value of kde)
            quant_prop: defines properties for the quantile lines ('thick, black')
    
    '''
    
    maxv = data.max()
    minv = data.min()
    meanv = data.mean()
    medianv = np.median(data)
    s_ys = np.array([minv, maxv, meanv, medianv])
    
    
    if not (quantiles == None):
        q_ys = np.zeros(len(quantiles))
        q_ys = np.percentile(data, 100*np.array(quantiles))
    
    points = kwargs.get('points', 100)
    
    bw = kwargs.get('bw', 'scott')
    kde = gaussian_kde(data, bw_method=bw)
    bw_used = kde.factor * data.std()
    
    if overshoot:
        cut_bottom = kwargs.get('cut_bottom', 2.0)
        cut_top = kwargs.get('cut_top', 2.0)
        span = maxv - minv
        ys = np.linspace(minv-bw_used*cut_bottom, maxv+bw_used*cut_top, points, endpoint=True)
    else:
        ys = np.linspace(minv, maxv, points, endpoint=True)
    
    xs = kde.evaluate(ys)
    s_xs = kde.evaluate(s_ys)
    
    if not (quantiles == None):
        q_xs = kde.evaluate(q_ys)
    
    
    #normalize
    normalize_fac = kwargs.get('normalize_fac', 1.0)
    if normalize == 'area':
        dy = ys[1] - ys[0]
        # trapezoidal rule
        if abs_data:
            # only consider data points that are postive for the scaling of the area
            xss = xs[ys>0]
            area = ((xss[1:] + xss[:-1])/2).sum()*dy
        else:
            area = ((xs[1:] + xs[:-1])/2).sum()*dy
        fac = normalize_fac / area
        xs = xs * fac
        s_xs = s_xs * fac
        if not (quantiles == None):
            q_xs = q_xs * fac
        
    if normalize == 'max':
        fac = normalize_fac / xs.max()
        xs = xs * fac
        s_xs = s_xs * fac
        if not (quantiles == None):
            q_xs = q_xs * fac
        
    flat_top_bot = kwargs.get('flat_top_bot', False)
    if flat_top_bot == False:
        plot_prop = plot_prop + ' cycle'
                
    print(f"\\fill [{fillcolor}] plot [{plot_prop}] coordinates ", end='{')
    # if violin plot is not complete use no 'circle' in plot_prop
    for x,y in zip(xs, ys):
        # shift to center
        x += center
        print(f"({x:.3f}, {y:.3f}) ", end='')
    
    if flat_top_bot:
        print('}'+ f' -- plot [{plot_prop}] coordinates ' + '{', end='')
        
    for x,y in zip(np.flip(-xs), np.flip(ys)):
        # shift to center
        x += center
        print(f"({x:.3f}, {y:.3f}) ", end='')
        
    if flat_top_bot:
        print('}' + f' -- ({xs[0]+center:.3f}, {ys[0]:.3f});')
    else:
        print('};')
    
    if draw_min_max:
        # draw max
        max_width = kwargs.get('max_width', s_xs[1]*2)
        max_prop = kwargs.get('max_prop', 'thick, black')
        print(f"\draw[{max_prop}] ({center - max_width/2:.3f}, {maxv:.3f}) -- ({center + max_width/2:.3f}, {maxv:.3f});  % max")
        # draw min
        min_width = kwargs.get('min_width', s_xs[0]*2)
        min_prop = kwargs.get('min_prop', 'thick, black')
        print(f"\draw[{min_prop}] ({center - min_width/2:.3f}, {minv:.3f}) -- ({center + min_width/2:.3f}, {minv:.3f});  % min")
        #draw vertical line
        vert_prop = kwargs.get('vert_prop', 'thick, black')
        print(f"\draw[{vert_prop}] ({center}, {minv:.3f}) -- ({center}, {maxv:.3f});  % vertical line")
    
    if draw_mean:
        mean_width = kwargs.get('mean_width', s_xs[2]*2)
        mean_prop = kwargs.get('mean_prop', 'thick, black')
        print(f"\draw[{mean_prop}] ({center - mean_width/2:.3f}, {meanv:.3f}) -- ({center + mean_width/2:.3f}, {meanv:.3f});  % mean")
    
    if draw_median:
        median_width = kwargs.get('median_width', s_xs[3]*2)
        median_prop = kwargs.get('median_prop', 'thick, black, dashed')
        print(f"\draw[{median_prop}] ({center - median_width/2:.3f}, {medianv:.3f}) -- ({center + median_width/2:.3f}, {medianv:.3f});  % median")
    
    if draw_quantiles:
        for idx, q in enumerate(q_xs):
            quant_width = kwargs.get('quant_width', q_xs[idx]*2)
            quant_prop = kwargs.get('quant_prop', 'thick, red')
            print(f"\draw[{quant_prop}] ({center - quant_width/2:.3f}, {q_ys[idx]:.3f}) -- ({center + quant_width/2:.3f}, {q_ys[idx]:.3f});  % quantiles {quantiles[idx]}")
    
    print_full = kwargs.get('print_full', False)
    if print_full:
        print('\n\n ### ADDITIONAL INFO #### \n\n')
        print('% this defines the pic')
        print('''
        \\begin{tikzpicture}[custom/.pic={\\fill[#1] (-0.2,-0.2) rectangle +(0.4, 0.4);}]]
            ''')
        print('% example legend, this beolngs in tikzpicture environment, but not in axis environment')
        print('''
        \\path (current bounding box.north east) node[matrix,anchor=north east,draw,nodes={anchor=center},inner sep=3pt, column sep=4pt]  {
        \\pic{custom=blue!40}; & \\node{test2};\\
        \\pic{custom=blue!40}; & \\node{test3};\\
        };
              ''')
        
