import argparse
import time

import matplotlib.pylab as plt
import numpy as np
from matplotlib.collections import LineCollection


def connections(start, current, order, n_hooks, collection, order_counter, new_tried):
    new_point = mod_no_zero(current * order[order_counter % len(order)], n_hooks)
    if new_point in collection:
        new_point = mod_no_zero(current * order[order_counter % len(order)] + 1, n_hooks)
        new_tried += 1

    if new_tried == n_hooks:
        return collection
    else:
        collection.append(new_point)
        return connections(start, new_point, order, n_hooks, collection, order_counter + 1, new_tried)


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


def make_cardioid(num_hooks, order=(2, 3), phase=(0, 0), offset=0):
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

    rescale_max = 60
    overall_max = 10

    order = [o / rescale_max * overall_max for o in order]

    check_mat = np.full((num_hooks, len(order)), False)
    cardioid = [[1, 1 * order[0]]]
    check_mat[0, 0] = True

    while not check_mat.all():
        for ind, this_order in enumerate(order):
            ph = phase[ind]
            last_point = int(cardioid[-1][-1])
            new_point = mod_no_zero(int(round(last_point * this_order + ph)),
                                    num_hooks)
            new_connection = [[last_point, new_point]]

            check = new_point == last_point
            check = check | (sorted(new_connection[0]) in [
                sorted(point) for point in cardioid])

            # if connection already exists
            if check:
                add = -1
                alread_checked = []
                new_point = mod_no_zero(last_point + add, num_hooks)
                while (sorted([last_point, new_point]) in [
                    sorted(point) for point in cardioid]) | (last_point == new_point):
                    if new_point in alread_checked:
                        add = -1
                        break
                    alread_checked.append(new_point)
                    add -= 1
                    new_point = mod_no_zero((last_point + add), num_hooks)

                new_connection = [[last_point,
                                   mod_no_zero((last_point + add),
                                               num_hooks)]]
            cardioid += new_connection
            check_mat[new_connection[0][0] - 1, ind] = True
            print("done: {:.1f}%".format(np.sum(check_mat) /
                                         check_mat.shape[0] /
                                         check_mat.shape[1] * 100))

    return [[int(point[0] + offset) % num_hooks, int(point[1] + offset) % num_hooks] for point in
            cardioid]


def plot_string_order(shape, string_order, fig=None, ax=None, color='#AC9F3C'):
    xlist = []
    ylist = []
    setup = False
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()
        setup = True
    fig.set_dpi(300)
    length_string = 0

    for ind, string_pair in enumerate(string_order):
        xlist.append(shape[string_pair[0], 0])
        ylist.append(shape[string_pair[0], 1])
        xlist.append(shape[string_pair[1], 0])
        ylist.append(shape[string_pair[1], 1])
        str_length = np.sum((shape[string_pair[0], :] -
                             shape[string_pair[1], :]) ** 2) ** 0.5 / 1000
        length_string += str_length

    coll1 = LineCollection(
        [np.column_stack([[x, x1], [y, y1]]) for x, x1, y, y1 in
         zip(xlist[:-1], xlist[1:], ylist[:-1], ylist[1:])],
        LineWidth=0.2,
        colors=[color] * (len(ylist) - 1))  # , cmap=plt.cm.jet)
    coll2 = LineCollection(
        [np.column_stack([[x, x1], [y, y1]]) for x, x1, y, y1 in
         zip(xlist[:-1], xlist[1:], ylist[:-1], ylist[1:])],
        linewidths=0.075,
        colors=['white'] * (len(ylist) - 1))

    coll1.set_array(np.asarray(range(len(string_order))))
    coll2.set_array(np.asarray(range(len(string_order))))
    ax.add_collection(coll1)
    ax.add_collection(coll2)
    if setup:
        ax.set_xlim(min(xlist), max(xlist))
        ax.set_ylim(min(ylist), max(ylist))

    rand_pair = np.random.randint(len(string_order))

    ax.plot([shape[string_order[rand_pair][0], 0], shape[string_order[rand_pair][1], 0]],
             [shape[string_order[rand_pair][0], 1], shape[string_order[rand_pair][1], 1]])
    if setup:
        ax.set_aspect('equal')
        ax.set_axis_off()

    return fig, ax


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-sa', '--save_as',
                        default='./results/nested_oscillations', type=str,
                        help='save path')

    parser.add_argument('-ho', '--hooks', default=500, type=int,
                        help='number of hooks (default=500)')

    parser.add_argument('-o', '--order', default='2,3', type=str,
                        help='order of nested oscillations (default=(2, 3))')

    parser.add_argument('-p', '--phase', default='o', type=str,
                        help='phase offset in hooks (default=(0,0))')

    parser.add_argument('-off', '--offset', default=0, type=int,
                        help='general offset in hooks (default=0)')

    parser.add_argument(
        '-sr', '--shape_rectangle',
        default='550,550', type=str,
        help='shape of rectangle a x b in mm (default=\'550, 550\')')

    parser.add_argument('-rc', '--radius_circle', default=400, type=int,
                        help='radius of circle in mm (default=400)')

    args = parser.parse_args()
    if args.phase == 'o':
        args.phase = ','.join(['0'] * len(args.order.split(',')))
    string_order = make_cardioid(args.hooks,
                                 order=tuple(map(
                                     float, args.order.split(','))),
                                 phase=tuple(map(
                                     float, args.phase.split(','))),
                                 offset=args.offset)

    args.save_as += '_ho-' + str(args.hooks)
    args.save_as += '_o-' + '-'.join(args.order.split(','))
    args.save_as += '_p-' + '-'.join(args.phase.split(','))
    args.save_as += '_off-' + str(args.offset)

    save_string_order(string_order, filename=args.save_as + '.csv')
    circle = make_circle(args.radius_circle, num_hooks=args.hooks)
    rectangle = make_rectangle(
        tuple(map(int, args.shape_rectangle.split(','))), num_hooks=args.hooks)

    for shape, name in zip((circle, rectangle), ('circle', 'rectangle')):
        # don't show circle
        if name == 'circle':
            continue
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
            str_length = np.sum((shape[string_pair[0], :] -
                                 shape[string_pair[1], :])**2)**0.5 / 1000
            print(str_length)
            length_string += str_length

        coll = LineCollection(
            [np.column_stack([[x, x1], [y, y1]]) for x, x1, y, y1 in
             zip(xlist[:-1], xlist[1:], ylist[:-1], ylist[1:])],
            LineWidth=0.1,
            colors=['b'] * (len(ylist) - 1))  # , cmap=plt.cm.jet)

        coll.set_array(np.asarray(range(len(string_order))))
        ax.add_collection(coll)
        ax.set_xlim(min(xlist), max(xlist))
        ax.set_ylim(min(ylist), max(ylist))

        rand_pair = np.random.randint(len(string_order))

        plt.plot([shape[string_order[rand_pair][0], 0], shape[string_order[rand_pair][1], 0]],
                 [shape[string_order[rand_pair][0], 1], shape[string_order[rand_pair][1], 1]])

        ax.set_aspect('equal')
        ax.set_axis_off()

        plt.savefig(args.save_as + '_' + name + '.svg', format='svg', dpi=1200)
        plt.savefig(args.save_as + '_' + name + '.pdf', format='pdf', dpi=1200)

        time.sleep(2)
        #plt.show()

        print('length string {} is {} m'.format(name, length_string))
        print('number of strings {} is {}'.format(name, len(string_order)))

        stats = np.asarray(string_order)
        stats = np.histogram(stats[:, 0], bins=range(args.hooks + 1))
        print('min strings hook {}'.format(stats[0].min()))
        print('median strings hook {}'.format(np.median(stats[0])))
        print('max strings hook {}'.format(stats[0].max()))

        fig, ax = plt.subplots()
        fig.set_dpi(600)
        ax.scatter(shape[:, 0], shape[:, 1], np.ones(shape[:, 0].shape) * 0.1)

        for i in range(len(shape)):
            angle = 0
            if shape[i, 1] >= (np.max([shape[:, 1]]) * (1 - 4 / len(shape))):
                angle = 90
            elif shape[i, 1] <= (np.min([shape[:, 1]]) * (1 - 4 / len(shape))):
                angle = -90

            if shape[i, 0] <= (np.min([shape[:, 0]]) * (1 - 4 / len(shape))):
                angle = 180
            ax.annotate(str(i), (shape[i, 0], shape[i, 1]),
                        rotation=angle,
                        rotation_mode='anchor',
                        fontsize=2,
                        verticalalignment='center')

        plt.plot(shape[:25, 0] + 100, shape[:25, 1] + 50)
        plt.text(np.mean(shape[:25, 0]) + 100, np.mean(shape[:25, 1]) + 52,
                 '10 cm')

        plt.plot([shape[string_order[rand_pair][0], 0], shape[string_order[rand_pair][1], 0]],
                 [shape[string_order[rand_pair][0], 1], shape[string_order[rand_pair][1], 1]])
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.savefig(args.save_as + '_' + name + '_holes.eps',
                    format='eps',
                    dpi=1200)
        time.sleep(2)
        #plt.show()

        fig, ax = plt.subplots()
        fig.set_dpi(600)

        mirror_shape = shape.dot([[-1,0],[0,1]]).astype(float)
        ax.scatter(mirror_shape[:, 0], mirror_shape[:, 1], np.ones(mirror_shape[:, 0].shape) * 0.1)

        for i in range(len(shape)):
            angle = 0
            if mirror_shape[i, 1] >= (np.max([mirror_shape[:, 1]]) * (1 - 4 / len(mirror_shape))):
                angle = 90
            elif mirror_shape[i, 1] <= (np.min([mirror_shape[:, 1]]) * (1 - 4 / len(mirror_shape))):
                angle = -90

            if mirror_shape[i, 0] <= (np.min([mirror_shape[:, 0]]) * (1 - 4 / len(mirror_shape))):
                angle = 180
            ax.annotate(' -' + str(i), (mirror_shape[i, 0], mirror_shape[i, 1]),
                        rotation=angle,
                        rotation_mode='anchor',
                        fontsize=2,
                        verticalalignment='center')

        plt.plot([mirror_shape[string_order[rand_pair][0], 0],
                  mirror_shape[string_order[rand_pair][1], 0]],
                 [mirror_shape[string_order[rand_pair][0], 1],
                  mirror_shape[string_order[rand_pair][1], 1]])

        plt.plot(mirror_shape[:25, 0] - 100, mirror_shape[:25, 1] + 50)
        plt.text(np.mean(mirror_shape[:25, 0]) - 100, np.mean(mirror_shape[:25, 1]) + 52,
                 '10 cm')
        ax.set_aspect('equal')
        ax.set_axis_off()
        plt.savefig(args.save_as + '_' + name + '_holes_mirror.eps',
                    format='eps',
                    dpi=1200)
        time.sleep(2)
        #plt.show()


if __name__ == "__main__":
    main()
