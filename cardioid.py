import argparse

import matplotlib.pylab as plt
import numpy as np
from matplotlib.collections import LineCollection


def save_string_order(array, filename):
    np.savetxt(filename, np.asarray([
        [ind, val[0], val[1]] for ind, val in enumerate(array)]),
               delimiter=',', fmt='%i')


def mod_no_zero(value, mod_value):
    return ((value - 1) % mod_value) + 1



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


def make_rectangle(rect_shape, num_hooks=250, offset=0):
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

    hooks_per_x = int(np.ceil(rect_shape[0] / (
            rect_shape[0] + rect_shape[1]) * num_hooks / 2))
    hooks_per_y = int(np.ceil(rect_shape[1] / (
            rect_shape[0] + rect_shape[1]) * num_hooks / 2))

    first_edge_x = np.linspace(0.0, rect_shape[0], hooks_per_x + 1)

    first_edge_y = np.linspace(0.0, rect_shape[1], hooks_per_y + 1)

    hooks_per_x2 = int(np.ceil(
        rect_shape[0] / (rect_shape[0] + rect_shape[1]) * (
                    num_hooks - hooks_per_x - hooks_per_y)))
    hooks_per_y2 = num_hooks - hooks_per_x - hooks_per_y - hooks_per_x2

    second_edge_x = np.linspace(0.0, rect_shape[0], hooks_per_x2 + 1)
    second_edge_y = np.linspace(0.0, rect_shape[1], hooks_per_y2 + 1)

    rectangle = [[x, 0] for x in first_edge_x]
    rectangle += [[first_edge_x[-1], y] for y in first_edge_y]
    rectangle += [[x, first_edge_y[-1]] for x in second_edge_x[::-1]]
    rectangle += [[0, y] for y in second_edge_y[::-1]]
    rectangle = rectangle[offset:] + rectangle[:offset]
    rectangle = np.asarray(rectangle)
    _, idx = np.unique(rectangle, return_index=True, axis=0)
    return rectangle[np.sort(idx)]


def make_cardioid(num_hooks, order=(2, 3)):
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
    :return: list of points indices (p, p + 1) that can be connected to form a
        cardioid shape in a concave set of points and number of hooks
        (which could change if `use_prime=True`)
    """

    check_mat = np.full((num_hooks, len(order)), False)
    cardioid = [[1, 1 * order[0]]]
    check_mat[0, 0] = True

    while not check_mat.all():
        for ind, this_order in enumerate(order):
            last_point = int(cardioid[-1][-1])
            new_connection = [[last_point, mod_no_zero((int(last_point * this_order)), num_hooks)]]
            if sorted(new_connection[0]) in [sorted(point) for point in cardioid]:
                if this_order == order[-1]:
                    add = 0
                    hook_counter = 0
                    while sorted([last_point, mod_no_zero((last_point + add), num_hooks)]) in [sorted(point) for point in cardioid]:
                        add += 1
                        hook_counter += 1
                        if hook_counter >= 10:
                            check_mat = np.full((num_hooks, len(order)), True)
                            break
                    new_connection = [[last_point, mod_no_zero((last_point + add), num_hooks)]]

            cardioid += new_connection
            check_mat[new_connection[0][0] - 1, ind] = True

    return [[int(point[0]) % num_hooks, int(point[1]) % num_hooks] for point in cardioid]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-sa', '--save_as',
                        default='./nested_oscillations', type=str,
                        help='save path')

    parser.add_argument('-ho', '--hooks', default=499, type=int,
                        help='number of hooks (default=499)')

    parser.add_argument('-o', '--order', default='2, 3', type=str,
                        help='order of nested oscillations (default=(2, 3))')

    parser.add_argument(
        '-sr', '--shape_rectangle',
        default='550,550', type=str,
        help='shape of rectangle a x b in mm (default=\'550, 550\')')

    parser.add_argument('-rc', '--radius_circle', default=400, type=int,
                        help='radius of circle in mm (default=400)')

    args = parser.parse_args()
    string_order = make_cardioid(args.hooks,
                                        order=tuple(map(
                                            float, args.order.split(','))))

    save_string_order(string_order, filename=args.save_as + '.csv')
    circle = make_circle(args.radius_circle, num_hooks=args.hooks)
    rectangle = make_rectangle(
        tuple(map(int, args.shape_rectangle.split(','))), num_hooks=args.hooks)

    for shape, name in zip((circle, rectangle), ('circle', 'rectangle')):
        xlist = []
        ylist = []

        fig, ax = plt.subplots()
        fig.set_dpi(300)
        length_string = 0

        for ind, string_pair in enumerate(string_order):
            xlist.append(shape[string_pair[0], 0])
            ylist.append(shape[string_pair[0], 1])
            xlist.append(shape[string_pair[1], 0])
            ylist.append(shape[string_pair[1], 1])
            length_string += np.linalg.norm(
                shape[string_pair[0], :] - shape[string_pair[1], :]) / 1000

        coll = LineCollection(
            [np.column_stack([[x, x1], [y, y1]]) for x, x1, y, y1 in
             zip(xlist[:-1], xlist[1:], ylist[:-1], ylist[1:])],
            LineWidth=0.1, colors=['b']*(len(ylist)-1))#, cmap=plt.cm.jet)

        coll.set_array(np.asarray(range(len(string_order))))
        ax.add_collection(coll)
        ax.set_xlim(min(xlist), max(xlist))
        ax.set_ylim(min(ylist), max(ylist))
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.savefig(args.save_as + '_' + name + '.svg', format='svg', dpi=1200)
        plt.show()

        print('length string {} is {} m'.format(name, length_string))


if __name__ == "__main__":
    main()
