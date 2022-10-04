import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t


cpdef cross_area(list args):
    cdef:
        double lf
        double df
        double ts_coef
        double theta_front
        double theta_back
        double fuselage_surface_area
        list upper_cross_area_dist
        list lower_cross_area_dist
        int x_num
        np.ndarray[DOUBLE_t, ndim=1] x_range
        double x_step
        double r_cross
        double xi
        double zmax
        double zmin
        double delta_surface_area
        np.ndarray[DOUBLE_t, ndim=1] y_range
        double yi
        double zu
        double zl

    lf, df, ts_coef, theta_front, theta_back, fuselage_surface_area = args

    # cross area distribution
    upper_cross_area_dist = []
    lower_cross_area_dist = []

    x_num = 200
    x_range = np.linspace(0, lf, x_num)
    x_step = (lf - 0) / x_num

    r_cross = 0.5 * df * ts_coef

    for xi in x_range:
        # define outside shape
        if 0 <= xi <= r_cross / np.tan(theta_front * np.pi / 180.0):
            zmax = r_cross * np.sqrt(1.0 - (xi - r_cross / np.tan(theta_front * np.pi / 180.0)) ** 2 / (r_cross / np.tan(theta_front * np.pi / 180.0)) ** 2)
            zmin = -zmax
            delta_surface_area = calc_eclipse_around_length([zmax, zmax / ts_coef])

        elif r_cross / np.tan(theta_front * np.pi / 180.0) <= xi <= lf - r_cross * 2 / np.tan(theta_back * np.pi / 180.0):
            zmax = r_cross
            zmin = -zmax
            delta_surface_area = calc_eclipse_around_length([zmax, zmax /ts_coef])

        else:
            zmin = r_cross - r_cross * 2 * np.sqrt(1.0 - (xi - (lf - 2 * r_cross / np.tan(theta_back * np.pi / 180.0))) ** 2 / (2 * r_cross / np.tan(theta_back * np.pi / 180.0)) ** 2)
            zmax = r_cross
            delta_surface_area = calc_eclipse_around_length([zmax, zmax / ts_coef])
            delta_surface_area *= 0.5

        # calculate surface area
        # print('z_min:',zmin, 'z_max:', zmax)
        fuselage_surface_area += delta_surface_area

        # range of y coords
        y_range = np.linspace(-zmax / ts_coef, zmax / ts_coef, 100)

        for yi in y_range:
            if zmax == 0:
                zu, zl = 0.0, 0.0
            else:
                zu = zmax * np.sqrt(1.0 - yi ** 2 / (zmax / ts_coef) ** 2)
                zl = zmin * np.sqrt(1.0 - yi ** 2 / (zmax / ts_coef) ** 2)

            upper_cross_area_dist.append([xi, yi, zu])
            lower_cross_area_dist.append([xi, yi, zl])

    # print('fuselage surface area:', fuselage_surface_area, 'theta back:', theta_back)


    results = [upper_cross_area_dist, lower_cross_area_dist, fuselage_surface_area]

    return results

cdef calc_eclipse_around_length(list eclipse_coefs):
    """

    :param eclipse_coefs:
    :return:
    """
    cdef:
        double target_eclipse_around_length
        double ac_coef
        double bc_coef
        double eps_c
        double theta
        double max_theta
        int calc_theta_num
        double theta_step
        int i
        double target_length


    # Initialize target length
    target_eclipse_around_length = 0
    ac_coef, bc_coef = eclipse_coefs

    if ac_coef == 0:
        return target_eclipse_around_length
    eps_c = (bc_coef / ac_coef) ** 2

    # print('ac_coef:', ac_coef, 'bc_coef:', bc_coef)

    if np.isnan(eps_c) or eps_c == np.inf:

        return target_eclipse_around_length

    # change pole angle
    theta = 0
    max_theta = np.pi * 0.5 / 180.0
    calc_theta_num = 20
    theta_step = (max_theta - 0) / calc_theta_num

    for i in range(calc_theta_num):
        target_length = 4 * ac_coef * np.sqrt(1.0 + eps_c * np.cos(theta) ** 2) * theta_step * 180.0
        target_eclipse_around_length += target_length
        # print('target_length:', target_length, 'theta:', theta)

        # update pole angle
        theta += theta_step

    return target_eclipse_around_length


cpdef hori_wing(list args):
    cdef:
        list upper_wing_area_dists
        list lower_wing_area_dists
        double wing_mounting_position
        double retreating_angle
        double wing_width
        double y_init
        list airfoil_args
        double croot
        double ctip
        double troot
        double ttip
        np.ndarray[DOUBLE_t, ndim=1] x_range
        list yupper
        list ylower
        double xi
        double ymin
        double ymax
        double yumin
        double yumax
        double ylmin
        double ylmax
        double yui
        double yli
        double slope
        double zu
        double zl
        list results

    upper_wing_area_dists, lower_wing_area_dists, wing_mounting_position, retreating_angle, wing_width, y_init, airfoil_args = args

    # set the airfoil arguments
    croot, ctip, troot, ttip = airfoil_args
    x_range = np.linspace(wing_mounting_position,wing_mounting_position + ctip + wing_width * np.tan(retreating_angle * np.pi / 180.0), 60)

    # y_init = df * 0.5 main wing

    yupper = []
    ylower = []

    for xi in x_range:
        # lower y coordinate
        if wing_mounting_position <= xi <= wing_mounting_position + croot:
            ymin = y_init
        else:
            ymin = wing_width / (wing_width * np.tan(retreating_angle * np.pi / 180.0) + ctip - croot) * \
                       (xi - croot - wing_mounting_position) + y_init

        # upper y coordinate
        if wing_mounting_position <= xi <= wing_mounting_position + wing_width * np.tan(
                    retreating_angle * np.pi / 180.0):
            ymax = np.tan((90 - retreating_angle) * np.pi / 180.0) * (xi - wing_mounting_position) + y_init

        else:
            ymax = wing_width + y_init

        yupper.append(ymax)
        ylower.append(ymin)

    # upper
    yumax, yumin = max(yupper), min(yupper)
    ylmax, ylmin = max(ylower), min(ylower)
    for xi, yui, yli in zip(x_range, yupper, ylower):
        slope = -0.5 * (troot - ttip) / wing_width
        zu = slope * (yui - yumin) + 0.5 * troot
        zl = slope * (yli - ylmin) + 0.5 * troot
        upper_wing_area_dists.append([xi, yui, zu])
        upper_wing_area_dists.append([xi, yli, zl])

        # opposite
        upper_wing_area_dists.append([xi, -yui, zu])
        upper_wing_area_dists.append([xi, -yli, zl])

        lower_wing_area_dists.append([xi, yui, -zu])
        lower_wing_area_dists.append([xi, yli, -zl])

        # opposite
        lower_wing_area_dists.append([xi, -yui, -zu])
        lower_wing_area_dists.append([xi, -yli, -zl])


    results = [upper_wing_area_dists, lower_wing_area_dists]

    return results


cpdef vert_wing(list args):
    cdef:
        list upper_wing_area_dists
        double wing_mounting_positions
        double retreating_angle
        double wing_width
        double z_init
        list airfoil_args
        double croot
        double ctip
        double troot
        double ttip
        np.ndarray[DOUBLE_t, ndim=1] x_range
        list zupper
        list zlower
        double xi
        double zmin
        double zmax
        double zumax
        double zumin
        double zlmax
        double zlmin
        double zui
        double zli
        double slope
        double yu
        double yl

    upper_wing_area_dists, wing_mounting_position, retreating_angle, wing_width, z_init, airfoil_args = args

    croot, ctip, troot, ttip = airfoil_args

    x_range = np.linspace(wing_mounting_position,
                              wing_mounting_position + ctip + wing_width * np.tan(
                                  retreating_angle * np.pi / 180.0), 60)

    zupper = []
    zlower = []

    for xi in x_range:
        # lower y coordinate
        if wing_mounting_position <= xi <= wing_mounting_position + croot:
            zmin = z_init
        else:
            zmin = wing_width / (wing_width * np.tan(retreating_angle * np.pi / 180.0) + ctip - croot) * \
                       (xi - croot - wing_mounting_position) + z_init

        # upper y coordinate
        if wing_mounting_position <= xi <= wing_mounting_position + wing_width * np.tan(
                    retreating_angle * np.pi / 180.0):
            zmax = np.tan((90 - retreating_angle) * np.pi / 180.0) * (xi - wing_mounting_position) + z_init

        else:
            zmax = wing_width + z_init

        zupper.append(zmax)
        zlower.append(zmin)

    # upper
    zumax, zumin = max(zupper), min(zupper)
    zlmax, zlmin = max(zlower), min(zlower)

    for xi, zui, zli in zip(x_range, zupper, zlower):
        slope = -0.5 * (troot - ttip) / wing_width
        yu = slope * (zui - zumin) + 0.5 * troot
        yl = slope * (zli - zlmin) + 0.5 * troot
        upper_wing_area_dists.append([xi, yu, zui])
        upper_wing_area_dists.append([xi, yl, zli])

        # opposite
        upper_wing_area_dists.append([xi, -yu, zui])
        upper_wing_area_dists.append([xi, -yl, zli])


    return upper_wing_area_dists

# engine
cpdef engine(list args):
    cdef:
        double engine_mounting_coefx
        double engine_mounting_coefy
        double BW
        double df
        double main_croot
        double main_troot
        double main_wing_mounting_point
        double m_to_ft
        list upper_engine_area_dists
        list lower_engine_area_dists
        double engine_mounting_position_y
        double engine_mounting_position_x
        int front_index
        double diam_in
        double diam_out
        int x_num
        np.ndarray[DOUBLE_t, ndim=1] x_range
        double xi
        double zmax
        np.ndarray[DOUBLE_t, ndim=1] y_range
        double yi
        double z
        double y1
        double y2
        double z1
        double z2
        list results

    engine_mounting_coefx, engine_mounting_coefy, BW, df, main_croot, main_troot, main_wing_mounting_point, engine_weight_class, m_to_ft = args

    upper_engine_area_dists = []
    lower_engine_area_dists = []

    # mounting position
    engine_mounting_position_y = engine_mounting_coefy * (0.5 * BW) + 0.5 * df
    engine_mounting_position_x = engine_mounting_coefx * main_croot + main_wing_mounting_point

    # calculate inlet diameter and out diameter
    if engine_weight_class.inlet_diameter[0, 10] <= 0.0:
        front_index = 20
    else:
        front_index = 10

    diam_in = engine_weight_class.inlet_diameter[0, front_index] * m_to_ft
    diam_out = np.sqrt(4.0 / np.pi * engine_weight_class.qref[2, 90]) * m_to_ft

    # print(engine_weight_class.total_engine_length * m_to_ft)
    # print('engine x:', engine_mounting_position_x)
    # print('engine y:', engine_mounting_position_y)

    x_num = 100
    x_range = np.linspace(0, engine_weight_class.total_engine_length * m_to_ft, x_num)

    for xi in x_range:
        zmax = 0.5 * (diam_in - diam_out) / (engine_weight_class.total_engine_length * m_to_ft) ** 2 + 0.5 * diam_in
        xi += engine_mounting_position_x
        # shape:circle
        y_range = np.linspace(-zmax, zmax, 50)

        for yi in y_range:

            z = np.sqrt(zmax ** 2 - yi ** 2)
            y1 = yi + engine_mounting_position_y
            y2 = -yi - engine_mounting_position_y
            z1 = z - main_troot - 0.5 * diam_in
            z2 = -z - main_troot - 0.5 * diam_in

            upper_engine_area_dists.append([xi, y1, z1])
            lower_engine_area_dists.append([xi, y1, z2])

            upper_engine_area_dists.append([xi, y2, z1])
            lower_engine_area_dists.append([xi, y2, z2])

    results = [upper_engine_area_dists, lower_engine_area_dists, engine_mounting_position_x, engine_mounting_position_y]

    return results

# distributed fan
cpdef distributed_fan(list args):
    cdef:
        double engine_mounting_coefx
        double engine_mounting_coefy
        double BW
        double df
        double main_croot
        double main_troot
        double theta
        double main_wing_mounting_point
        double m_to_ft
        str distfan_mount
        list upper_distributed_fan_dists
        list lower_distributd_fan_dists
        double df_diam_in
        double df_diam_out
        int Nfan
        double distributed_engine_coefx
        double distributed_engine_coefy
        double distributed_fan_mounting_positions_x
        double distributed_fan_mounting_positions_y
        double rect_length
        int n
        int x_num
        np.ndarray[DOUBLE_t, ndim=1] x_range
        double xi
        double zmax
        np.ndarray[DOUBLE_t, ndim=1] y_range
        double yi
        double z
        double y1
        double y2
        double z1
        double z2

    engine_mounting_coefx, engine_mounting_coefy, BW, df, main_croot, main_troot, theta, main_wing_mounting_point, engine_weight_class, m_to_ft, distfan_mount = args

    upper_distributed_fan_dists = []
    lower_distirbuted_fan_dists = []

    df_diam_in = engine_weight_class.fan_in_diameter[0] * m_to_ft
    df_diam_out = engine_weight_class.fan_out_diameter[0] * m_to_ft

    # fan number
    Nfan = int(engine_weight_class.Nfan)

    distributed_engine_coefx = 0.1  # 0.2 (upper)
    distributed_engine_coefy = 0.1  # 0.1 (upper)

    if distfan_mount == 'below':
        distributed_engine_coefx += engine_mounting_coefx
        distributed_engine_coefy += engine_mounting_coefy

    distributed_fan_mounting_positons_y = distributed_engine_coefy * (0.5 * BW) + df * 0.5
    distributed_fan_mounting_positons_x = distributed_engine_coefx * main_croot + main_wing_mounting_point
    rect_length = 3

    # calculate distributed fan coordinates
    for n in range(Nfan):

        x_num = 100
        x_range = np.linspace(0, engine_weight_class.distributed_fan_length * m_to_ft, x_num)

        for xi in x_range:
            zmax = 0.5 * (df_diam_in - df_diam_out) / (engine_weight_class.distributed_fan_length * m_to_ft) ** 2 + 0.5 * df_diam_in
            xi += distributed_fan_mounting_positons_x
            # shape:circle
            y_range = np.linspace(-zmax, zmax, 50)

            for yi in y_range:

                z = np.sqrt(zmax ** 2 - yi ** 2)
                y1 = yi + distributed_fan_mounting_positons_y
                y2 = -yi - distributed_fan_mounting_positons_y
                if distfan_mount == 'below':
                    z1 = z - main_troot - 0.5 * df_diam_in
                    z2 = -z - main_troot - 0.5 * df_diam_in
                else:
                    z1 = z + main_troot + 0.5 * df_diam_in
                    z2 = -z + main_troot + 0.5 * df_diam_in

                upper_distributed_fan_dists.append([xi, y1, z1])
                lower_distirbuted_fan_dists.append([xi, y1, z2])

                upper_distributed_fan_dists.append([xi, y2, z1])
                lower_distirbuted_fan_dists.append([xi, y2, z2])

        # update distributed fan positions
        distributed_fan_mounting_positons_y += 1.1 * df_diam_in
        distributed_fan_mounting_positons_x += rect_length * np.cos(theta * np.pi / 180.0)



    results = [upper_distributed_fan_dists, lower_distirbuted_fan_dists, distributed_fan_mounting_positons_x, distributed_fan_mounting_positons_y]

    return results

# shelter
cpdef shelter(args):
    cdef:
        list upper_distributed_fan_dists
        double df
        double main_troot
        double theta
        double main_wing_mounting_point
        double m_to_ft
        list upper_shelter_dists
        list lower_shelter_dists
        double zmax
        double Nfan
        double ymax
        double xmax
        np.ndarray[DOUBLE_t, ndim=1] xrange
        double angle
        double ylmax
        double coef
        double xi
        double yu
        double yl
        np.ndarray[DOUBLE_t, ndim=1] yrange
        double yi
        double zu
        double zl
        double xu
        list results
    upper_distributed_fan_dists, df, main_troot, theta, main_wing_mounting_point, engine_weight_class, m_to_ft = args

    upper_shelter_dists = []
    lower_shelter_dists = []

    zmax = np.max(np.array(upper_distributed_fan_dists)[:, 2]) * 1.1
    Nfan = engine_weight_class.Nfan
    ymax = engine_weight_class.distributed_fan_width_length * m_to_ft * (Nfan * 0.7)
    xmax = engine_weight_class.distributed_fan_length * m_to_ft * 2.0

    xrange = np.linspace(0, xmax, 100)

    # angle
    angle = 90 - theta
    ylmax = ymax * 1.1
    coef = (Nfan * 0.7) * 1.1
    for xi in xrange:
        if 0 <= xi <= ylmax / np.tan(angle * np.pi / 180.0) / coef:
            yu = df * 0.5 + xi * np.tan(angle * np.pi / 180.0)
            yl = yu

        else:
            yl = ylmax
            yu = ymax

        yrange = np.linspace(df * 0.5, yl, 30)

        for yi in yrange:
            if df * 0.5 <= yi <= yu:
                zu = zmax
                zl = main_troot
            else:
                zu = -zmax / (ylmax - ymax) * (yi - ymax) + zmax
                zl = main_troot

            xu = xi + main_wing_mounting_point
            upper_shelter_dists.append([xu, yi, zu])
            upper_shelter_dists.append([xu, -yi, zu])

            lower_shelter_dists.append([xu, yi, zl])
            lower_shelter_dists.append([xu, -yi, zl])

    results = [upper_shelter_dists, lower_shelter_dists]

    return results



################ Blended Wing Body ################

# NACA5 type
cdef airfoil_type_5(double x_chord):
    cdef double m
    cdef double k1
    cdef double yc
    # 5 airfoil
    # place of crossing front and back chamber line
    m = 0.2025  # representative value (230)
    k1 = 15.957  # attack of angle(AoA) at design point

    # camber line shape
    if x_chord < m:

        yc = (1 / 6) * k1 * (x_chord ** 3 - 3 * m * x_chord ** 2 + m ** 2 * (3 - m) * x_chord)

    else:

        yc = (1 / 6) * k1 * m ** 3 * (1.0 - x_chord)

    return yc

# NACA4 type
cdef airfoil_type_4(double x_chord):
    cdef float m
    cdef float p
    cdef double yc
    # 4 airfoil
    m = 0.02
    p = 0.4
    if x_chord < p:

        yc = (m / p ** 2) * (2 * p * x_chord - x_chord ** 2)

    else:

        yc = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x_chord - x_chord ** 2)

    return yc


cpdef calc_airfoil(list args):
    cdef double x_chord
    cdef int airfoil_type
    cdef double tc
    cdef double tcroot

    x_chord, airfoil_type, tc ,tcroot = args

    # calculate camber line
    yc = 0
    if airfoil_type == 4:
        yc = airfoil_type_4(x_chord)
    elif airfoil_type == 5:
        yc = airfoil_type_5(x_chord)

    # distribution of thickness and radius of leading edge
    # tc = .tc
    tc = tc * 1.07
    yt = tc / 0.2 * (0.29690 * np.sqrt(x_chord) - 0.126 * x_chord - 0.3516 * x_chord ** 2 +
                              0.28430 * x_chord ** 3 - 0.1015 * x_chord ** 4)
    rt = 1.1019 * tcroot ** 2

    return yc, yt, rt

cpdef eclipse_around_length_bwb(list args):
    cdef:
        double eps_u
        double eps_l
        double au_coef
        double al_coef
        double overall_chord
        double eclipse_ar_length
        double theta
        double delta_theta
        double max_theta
        int theta_step
        int th_step
        double target_length

    eps_u, eps_l, au_coef, al_coef, overall_chord = args
    eclipse_ar_length = 0.0
    theta = 0.0
    delta_theta = 0.01 * np.pi / 180.0
    max_theta = np.pi * 0.5 / 180.0
    # step of summing up function value
    theta_step = int(max_theta / delta_theta)

    for th_step in range(theta_step):
        target_length = 0.5 * (np.sqrt(1.0 + eps_u ** 2 * np.cos(theta) ** 2) * 4 * au_coef + 4 * al_coef * np.sqrt(
                1.0 + eps_l ** 2 * np.cos(theta) ** 2))
        eclipse_ar_length += target_length * delta_theta * overall_chord
        # print(eps_u, theta, eps_l)
        # print('target eclipse length:', target_length)
        # update angle
        theta += delta_theta

    return eclipse_ar_length


cpdef calc_cabin_outside_shape(list args):
    cdef:
        double bubl
        double delta_x
        double tc
        double tcroot
        list cabin_outshape_datas
        double dist_fan_point
        double overall_chord
        double cabin_expand_coef
        double top_wet_area
        double side_wet_area
        list delta_eclipse_surface_area
        list delta_eclipse_volume
        list upper_cross_area_dist
        list lower_cross_area_dist
        list outside_shpe
        double xi
        double yi
        double x_chord
        double z_up
        double zinit
        double au
        double bu
        double al
        double bl
        double cross_ar_length
        double eps_u
        double eps_l
        double cross_area
        np.ndarray yrange
        double yu
        double z_u
        double z_l
        double x_real
        double y_real
        double z_u_real
        double z_l_real
        list results

    bubl, delta_x, tc, tcroot, cabin_outshape_datas, dist_fan_point, overall_chord, cabin_expand_coef, top_wet_area, side_wet_area, delta_eclipse_surface_area, delta_eclipse_volume, upper_cross_area_dist, lower_cross_area_dist = args

    # for mass calculation
    outside_shape = []

    for xi, yi in cabin_outshape_datas:
        x_chord = xi / overall_chord
        args = [x_chord, 4, tc, tcroot]
        _, z_up, _ = calc_airfoil(args)

        if xi == dist_fan_point:
            zinit = z_up * overall_chord * cabin_expand_coef

        # eclipse length
        au = yi
        bu = z_up * overall_chord * cabin_expand_coef

        al = au
        bl = bu / bubl


        if au == 0 or al == 0:
            cross_ar_length = 0
        else:
            eps_u = (bu / au)
            eps_l = (bl / al)
            args = [eps_u, eps_l, au, al, overall_chord]

            cross_ar_length = eclipse_around_length_bwb(args)


        # for mass calculation
        outside_shape.append([au, bu, bl])

        cross_area = 0.5 * np.pi * (au * bu + al * bl)

        top_wet_area += 2 * (yi * delta_x * overall_chord)
        side_wet_area += (z_up + z_up / bubl) * overall_chord * delta_x * overall_chord

        delta_eclipse_surface_area.append(cross_ar_length * delta_x * overall_chord)
        delta_eclipse_volume.append(cross_area * delta_x * overall_chord)

        yrange = np.linspace(-al, al, 50)

        for yu in yrange:
            # Range of y coords
            if au == 0 or al == 0:
                z_u = 0
                z_l = 0
            else:

                z_u = bu * np.sqrt(1.0 - yu ** 2 / au ** 2)
                z_l = -bl * np.sqrt(1.0 - yu ** 2 / al ** 2)

            # change real coordinates
            x_real = xi
            y_real = yu
            z_u_real = z_u
            z_l_real = z_l

            upper_cross_area_dist.append([x_real, y_real, z_u_real])
            lower_cross_area_dist.append([x_real, -y_real, z_l_real])


    results = [top_wet_area, side_wet_area, delta_eclipse_surface_area, delta_eclipse_volume, upper_cross_area_dist, lower_cross_area_dist, outside_shape]
    return results

cpdef bwb_main_wing(list args):
    cdef:
        double wing_mounting_position
        double main_ctip
        double main_croot
        double main_troot
        double main_ttip
        double bwb_BW
        double bwb_df
        double theta
        list upper_wing_area_dist
        list lower_wing_area_dist
        np.ndarray x_range
        list yupper
        list ylower
        double xi
        double ymin
        double ymax
        double yumax
        double yumin
        double ylmax
        double ylmin
        double yui
        double yli
        double slope
        double zu
        double zl
        list results

    wing_mounting_position, main_ctip, main_croot, main_troot, main_ttip, bwb_BW, bwb_df, theta, upper_wing_area_dist, lower_wing_area_dist = args

    x_range = np.linspace(wing_mounting_position, wing_mounting_position + main_ctip + bwb_BW * np.tan(theta * np.pi / 180.0), 40)

    y_init = bwb_df * 0.5
    yupper = []
    ylower = []

    for xi in x_range:
        # lower y coordinate
        if wing_mounting_position <= xi <= wing_mounting_position + main_croot:
            ymin = y_init
        else:
            ymin = bwb_BW / (bwb_BW * np.tan(theta * np.pi / 180.0) + main_ctip - main_croot) * \
                       (xi - main_croot - wing_mounting_position) + y_init

        # upper y coordinate
        if wing_mounting_position <= xi <= wing_mounting_position + bwb_BW * np.tan(theta * np.pi / 180.0):
            ymax = np.tan((90 - theta) * np.pi / 180.0) * (xi - wing_mounting_position) + y_init

        else:
            ymax = bwb_BW + y_init

        yupper.append(ymax)
        ylower.append(ymin)

    # upper
    yumax, yumin = max(yupper), min(yupper)
    ylmax, ylmin = max(ylower), min(ylower)
    for xi, yui, yli in zip(x_range, yupper, ylower):
        slope = -0.5 * (main_troot - main_ttip) / bwb_BW
        zu = slope * (yui - yumin) + 0.5 * main_troot
        zl = slope * (yli - ylmin) + 0.5 * main_troot
        upper_wing_area_dist.append([xi, yui, zu])
        upper_wing_area_dist.append([xi, yli, zl])

        # opposite
        upper_wing_area_dist.append([xi, -yui, zu])
        upper_wing_area_dist.append([xi, -yli, zl])

        lower_wing_area_dist.append([xi, yui, -zu])
        lower_wing_area_dist.append([xi, yli, -zl])

        # opposite
        lower_wing_area_dist.append([xi, -yui, -zu])
        lower_wing_area_dist.append([xi, -yli, -zl])

    results = [upper_wing_area_dist, lower_wing_area_dist]
    return results


cpdef bwb_engine(args):
    cdef:
        engine_weight_class
        int front_index
        double main_troot
        double engine_mounting_position_x
        double engine_mounting_position_y
        list upper_engine_area_dists
        list lower_engine_area_dists
        double m_to_ft
        double diam_in
        double diam_out
        int x_num
        np.ndarray x_range
        double xi
        double zmax
        np.ndarray y_range
        double yi
        double z
        double y1
        double y2
        double z1
        double z2


    engine_weight_class, front_index, main_troot, engine_mounting_position_x, engine_mounting_position_y, upper_engine_area_dists, lower_engine_area_dists, m_to_ft = args

    diam_in = engine_weight_class.inlet_diameter[0, front_index] * m_to_ft
    diam_out = np.sqrt(4.0 / np.pi * engine_weight_class.qref[2, 90]) * m_to_ft


    x_num = 100
    x_range = np.linspace(0, engine_weight_class.total_engine_length * m_to_ft, x_num)

    for xi in x_range:
        zmax = 0.5 * (diam_in - diam_out) / (
                        engine_weight_class.total_engine_length * m_to_ft) ** 2 + 0.5 * diam_in
        xi += engine_mounting_position_x
        # shape:circle
        y_range = np.linspace(-zmax, zmax, 50)

        for yi in y_range:
            z = np.sqrt(zmax ** 2 - yi ** 2)
            y1 = yi + engine_mounting_position_y
            y2 = -yi - engine_mounting_position_y
            z1 = z - main_troot - 0.5 * diam_in
            z2 = -z - main_troot - 0.5 * diam_in

            upper_engine_area_dists.append([xi, y1, z1])
            lower_engine_area_dists.append([xi, y1, z2])

            upper_engine_area_dists.append([xi, y2, z1])
            lower_engine_area_dists.append([xi, y2, z2])

    results = [upper_engine_area_dists, lower_engine_area_dists]

    return results


cpdef bwb_distributed_fan(list args):
    cdef:
        engine_weight_class
        int Nfan
        double distributed_fan_mounting_positions_x
        double distributed_fan_mounting_positions_y
        double main_troot
        double zinit
        double df_diam_in
        double df_diam_out
        list upper_distributed_fan_dists
        list lower_distributed_fan_dists
        double m_to_ft
        int x_num
        int n
        np.ndarray x_range
        double xi
        double zmax
        np.ndarray y_range
        double yi
        double z
        double y1
        double y2
        double z1
        double z2

    engine_weight_class, Nfan, distributed_fan_mounting_positions_x, distributed_fan_mounting_positions_y, main_troot, zinit, df_diam_in, df_diam_out, upper_distributed_fan_dists, lower_distributed_fan_dists, m_to_ft = args
    x_num = 100
    # calculate distributed fan coordinates
    for n in range(Nfan):

        x_range = np.linspace(0, engine_weight_class.distributed_fan_length * m_to_ft, x_num)

        for xi in x_range:
            zmax = 0.5 * (df_diam_in - df_diam_out) / (engine_weight_class.distributed_fan_length * m_to_ft) ** 2 + 0.5 * df_diam_in
            xi += distributed_fan_mounting_positions_x
            # shape:circle
            y_range = np.linspace(-zmax, zmax, 50)

            for yi in y_range:

                z = np.sqrt(zmax ** 2 - yi ** 2)
                y1 = yi + distributed_fan_mounting_positions_y
                y2 = -yi - distributed_fan_mounting_positions_y
                z1 = z + main_troot + 0.5 * df_diam_in + zinit
                z2 = -z + main_troot + 0.5 * df_diam_in + zinit

                upper_distributed_fan_dists.append([xi, y1, z1])
                lower_distributed_fan_dists.append([xi, y1, z2])

                upper_distributed_fan_dists.append([xi, y2, z1])
                lower_distributed_fan_dists.append([xi, y2, z2])

        # update distributed fan positions
        distributed_fan_mounting_positions_y += 1.1 * df_diam_in

    results = [distributed_fan_mounting_positions_y, upper_distributed_fan_dists, lower_distributed_fan_dists]

    return results
