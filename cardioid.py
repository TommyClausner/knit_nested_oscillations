import argparse

import matplotlib.pylab as plt
import numpy as np


def save_string_order(array, filename):
    np.savetxt(filename, np.asarray([
        [ind, val[0], val[1]] for ind, val in enumerate(array)]),
               delimiter=',', fmt='%i')


def make_circle(radius, num_hooks=201):
    """
    Make circle coordinates.

    :param int|float radius:
        Radius of circle to span.
    :param int num_hooks:
        How many points are used to create the circle.
    :return: np.array of circle coordinates (x, y)
    """
    theta = np.linspace(0, 2 * np.pi, num_hooks)
    return np.round(np.asarray([
        (radius / 2 - 1) * np.cos(theta),
        (radius / 2 - 1) * np.sin(theta)]) +
                    radius / 2).astype(int).T


def make_rectangle(rect_shape=(1000, 1000), num_hooks=250, offset=0):
    """
    Make rectangle coordinates.

    :param tuple|list rect_shape:
        (width, height) of rectangle.
    :param int num_hooks:
        How many points are used to create the rectangle.
    :param int offset:
        Offset for point numbering (to shift motive around the rectangle)
    :return: np.array of rectangle coordinates (x, y)
    """
    delta = (rect_shape[0] * 2 + rect_shape[1] * 2) / num_hooks

    one_edge_x = np.arange(0.0, rect_shape[0], delta).astype(int)
    one_edge_y = np.arange(0.0, rect_shape[1], delta).astype(int)

    rectangle = [[x, 0] for x in one_edge_x]
    rectangle += [[rect_shape[0], y] for y in one_edge_y]
    rectangle += [[x, rect_shape[1]] for x in one_edge_x[::-1]]
    rectangle += [[0, y] for y in one_edge_y[::-1]]
    rectangle = rectangle[offset:] + rectangle[:offset]
    return np.asarray(rectangle)


def is_prime(number):
    """
    Check if number is prime

    :param int number:
    :return: True or False whether the numebr is prime
    """
    if number < 2:
        return False

    for num in range(2, number):
        if number % num == 0:
            return False
    return True


def nearest_prime(number):
    """
    Find nearest prime around number.

    :param int number:
    :return: int prime with lowest difference to number
    """
    sign = 1
    counter = 0
    while True:
        new_number = number + sign * counter
        if is_prime(new_number):
            break
        else:
            sign *= -1
        if sign == 1:
            counter += 1
    return new_number


def make_cardioid(num_hooks, order=(2, 3), min_strings=None, use_prime=True):
    """
    Make order to connect concave set of input points to cardioid shape. This
    will be achieved, by connecting points in the concave shape in the
    following way:

    - we start at (0, 1), which connects the first to the second point
    - we continue at (1, 1 * order[0]), which connects the second point to the
    4th point if order[0] is 2.
    - we continue at (1 * order[0], 1 * order[0] * order[1]), which connects
    the 4th point if order[0] is 2 with the 12th point if order[1] is 3.

    Hence we connect each point p to the corresponding point p * order, where p
    is the index of a coordinate in a set of num_hooks concave points.

    :param int num_hooks:
        Number of points to connect.
    :param tuple|list order:
        Orders to form the cardioid.
    :param None|int min_strings:
        Minimum number of connections. If None it will only be ensured, that
        each point is passed by at least once.
    :param bool use_prime:
        Whether to find the nearest prime number around `num_hooks` and replace
        `num_hooks` with it.
    :return: list of points indices (p, p + 1) that can be connected to form a
        cardioid shape in a concave set of points and number of hooks
        (which could change if `use_prime=True`)
    """

    # make number prime if desired
    if not is_prime(num_hooks):
        if use_prime:
            print('Prime number needed to create continuous string pattern...')
            print('Finding nearest prime...')
            num_hooks = nearest_prime(num_hooks)
            print('Using %d instead.' % num_hooks)
            print('To avoid this set use_prime=False.')

    # initialize computation
    cardioid = [(0, 1)]
    started = []
    check = [False]

    # minimum number of from connections per point
    min_val = 1
    while not all(check):
        print('left to check {}'.format(len(check) - np.sum(check)))

        # iterate through order and make point pairs
        for ordr in np.random.permutation(np.asarray(order)):
            pair = (cardioid[-1][1], int(cardioid[-1][1] * ordr % num_hooks))

            # if pair already present (we only allow unique connections) then
            # we just go to the next point in order, that does not already is
            # occupied by a pair
            shift = num_hooks
            regular_connection = True
            while (pair in cardioid) or (tuple(pair[::-1]) in cardioid):
                pair = (
                    cardioid[-1][1],
                    int((cardioid[-1][1] - shift) % num_hooks))

                # go to next point if pair already exist
                shift += 1
                regular_connection = False

            # append valid pair and list starting points only for
            # non-bypassing connections.
            if regular_connection:
                started.append(pair[0])
            cardioid.append(pair)

        # formulate check for while loop (if all connections are established)
        check = [np.count_nonzero(np.asarray(started) == val) >= min_val for
                 val in range(num_hooks)]

        # if a minimum number of strings is selected, add this to the check
        if min_strings is not None:
            # copy as many `False` to the check vector as strings are left
            # (we increment by two, since we only use the first n - 2 points)
            check += [False] * (min_strings - len(cardioid) + 2)
    return cardioid[:-2], num_hooks


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--strings', default=None, type=int,
                        help='number of precomputed strings (default=None)')

    parser.add_argument('-sa', '--save_as',
                        default='./nested_oscillations', type=str,
                        help='save path')

    parser.add_argument('-ho', '--hooks', default=499, type=int,
                        help='number of hooks (default=499)')

    parser.add_argument('-np', '--nprime', action="store_true",
                        help='do not find nearest prime')

    parser.add_argument('-o', '--order', default=(2, 3), type=tuple,
                        help='order of nested oscillations (default=(2, 3))')

    parser.add_argument(
        '-sr', '--shape_rectangle',
        default='400,400', type=str,
        help='shape of rectangle a x b in mm (default=\'400, 400\')')

    parser.add_argument('-rc', '--radius_circle', default=400, type=int,
                        help='radius of circle in mm (default=400)')

    args = parser.parse_args()
    string_order, hooks = make_cardioid(args.hooks, order=args.order,
                                        min_strings=args.strings,
                                        use_prime=not args.nprime)

    save_string_order(string_order, filename=args.save_as + '.csv')
    circle = make_circle(args.radius_circle, num_hooks=hooks)
    rectangle = make_rectangle(
        tuple(map(int, args.shape_rectangle.split(','))), num_hooks=hooks)

    for shape, name in zip((circle, rectangle), ('circle', 'rectangle')):
        xlist = []
        ylist = []
        plt.figure(dpi=1200)
        length_string = 0
        for ind, string_pair in enumerate(string_order):
            xlist.append(shape[string_pair[0], 0])
            ylist.append(shape[string_pair[0], 1])
            xlist.append(shape[string_pair[1], 0])
            ylist.append(shape[string_pair[1], 1])
            length_string += np.linalg.norm(
                shape[string_pair[0], :] - shape[string_pair[1], :]) / 1000
        plt.plot(xlist, ylist, 'black', LineWidth=0.01)
        plt.axis('equal')
        plt.axis('off')
        plt.show()
        plt.savefig(args.save_as + '_' + name + '.png')

        print('length string {} is {} m'.format(name, length_string))


if __name__ == "__main__":
    main()


def z_angle(width, wood_thickness, height, side_offset):
    """
    Computes the angle for a 'Z' between two parallel bars.

    :param float|int width:
        Width of the parallel bars.
    :param float|int wood_thickness:
        Thickness of the 'Z' - bar in parallel direction to the parallel bars.
    :param float|int height:
        Height from 'top-to-top' of the parallel bars.
    :param float|int side_offset:
        Offset from the sides of each parallel bar to the 'Z' - bar.
    :return: The sine of the angle in which the 'Z' - bar needs to be placed.
    """
    w = width
    t = wood_thickness
    h = height
    o = side_offset
    return (t * (w - 2 * o) + ((h ** 2 * (
            (w - 2 * o) ** 2 + h ** 2 - t ** 2) + 4 * t ** 2 * (
                                        o - o ** 2)) ** 0.5)) / (
                   h ** 2 + (w - 2 * o) ** 2)
