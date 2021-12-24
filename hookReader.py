import argparse
import collections
import os
import sys

import numpy as np
import speech_recognition as sr
from gtts import gTTS
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from cardioid import make_rectangle


def get_side(this_hook):
    if this_hook >= 375:
        return 3
    if this_hook >= 250:
        return 2
    if this_hook >= 125:
        return 1
    return 0


def get_skip_hooks(all_hooks):
    skip_hooks = [get_side(all_hooks[0]) == get_side(all_hooks[1])]

    corner_hooks = {0: [3, 0], 125: [0, 1], 250: [1, 2], 375: [2, 3]}

    for prev_hook, curr_hook, next_hook in zip(all_hooks[:-2],
                                               all_hooks[1:-1],
                                               all_hooks[2:]):
        prev_hook_side = get_side(prev_hook)
        curr_hook_side = get_side(curr_hook)
        next_hook_side = get_side(next_hook)

        if curr_hook in corner_hooks:
            corner_sides = corner_hooks[curr_hook]
            if prev_hook_side in corner_sides:
                curr_hook_side = prev_hook_side + 0

        skip_this = prev_hook_side == next_hook_side == curr_hook_side
        skip_hooks.append(skip_this)

    skip_hooks.append(get_side(all_hooks[-2]) == get_side(all_hooks[-1]))

    return skip_hooks


def read_string(string):
    speech = gTTS(text=string, lang='en', slow=False)
    tmpmp3 = "tmp.mp3"
    speech.save(tmpmp3)
    os.system("afplay " + tmpmp3)


class HookReader:
    all_hooks = None
    data = None
    data_path = ''
    current_numbers = [None, None, None]
    counter = 0
    fig = None
    ax = None
    status_plot = None
    key_control = None

    def __init__(self, data_path=None, compute_dir=None, overwrite=False, mic_id=0):
        if data_path is not None:
            self.data_path = data_path

        file = self.counter_file_name(compute_dir)
        if overwrite:
            os.remove(file)
            np.savetxt(file, [self.counter, ])
        else:
            if os.path.isfile(file):
                self.counter = np.genfromtxt(file).astype(int)

        self.compute_dir = compute_dir
        self.load_data()

        self.rectangle = collections.deque(make_rectangle((2400, 2400), 500).tolist())
        self.rectangle = np.asarray(self.rectangle)

        self.source = sr.Microphone(device_index=mic_id)
        self.r = sr.Recognizer()
        with self.source:
            self.r.adjust_for_ambient_noise(self.source)
            self.r.pause_threshold = 0.5
            self.r.non_speaking_duration = 0.5

    def on_press(self, event):
        if event.key == 'right' or event.key == 'n':
            self.counter += 1
        elif event.key == 'left' or event.key == 'p':
            self.counter -= 1
        else:
            return
        self._process_counter_state()

    def connect(self):
        self.key_control = self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def _process_counter_state(self):
        if self.counter < 0:
            self.counter = 0
        elif self.counter > (len(self.all_hooks) - 1):
            self.counter = len(self.all_hooks) - 1

        np.savetxt(self.counter_file_name(self.compute_dir), [self.counter, ])

        self.update_current_numbers()
        self.read_number()

    def counter_file_name(self, path=None):
        if path is None:
            path, file = os.path.dirname(self.data_path), os.path.basename(self.data_path)
        else:
            _, file = os.path.dirname(self.data_path), os.path.basename(self.data_path)
        return os.path.join(path, 'counter_' + file)

    def detect_keywords(self, _, audio):  # this is called from the background thread
        #keywords = [("next", 0), ("previous", 0), ("repeat", 0)]
        try:
            word = self.r.recognize_google(audio)#, keyword_entries=keywords).split()
            self.process_speech(word)
        except sr.UnknownValueError:
            print("Come again ...")

    def process_speech(self, speech_as_text):
        if speech_as_text is None:
            return

        if "next" in speech_as_text:
            self.counter += 1
        elif "previous" in speech_as_text:
            self.counter -= 1
        elif "repeat" in speech_as_text:
            self.read_number()
            return
        else:
            return

        self._process_counter_state()

    def update_current_numbers(self):
        self.current_numbers = [
            self.all_hooks[self.counter - 1] if self.counter > 0 else '',
            self.all_hooks[self.counter],
            self.all_hooks[self.counter + 1] if self.counter < (len(self.all_hooks) - 1) else ''
        ]

    def start(self):
        self.update_current_numbers()
        self.read_number()
        return self.r.listen_in_background(self.source, self.detect_keywords)

    def update_plot(self):
        string_order = self.data[:, 1:]
        shape = self.rectangle
        h = self.counter
        self.ax.plot([shape[string_order[h][0], 0], shape[string_order[h][1], 0]],
                     [shape[string_order[h][0], 1], shape[string_order[h][1], 1]], 'b', linewidth=.1)
        x, y = shape[string_order[h][1], 0], shape[string_order[h][1], 1]
        self.status_plot.set_offsets(np.c_[x, y])
        title = [val for val in self._make_output().split('\n')
                 if (len(val) > 0) & ('done' not in val) & ('#' not in val)]
        title = ' | '.join(title) + '\n\n'
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def _make_output(self):
        return "\n#####\ncurrent: {}\n\nnext: {}\nprevious: {}\ndone: {}/{}\n#####\n".format(
            self.current_numbers[1],
            self.current_numbers[2],
            self.current_numbers[0],
            self.counter + 1,
            len(self.all_hooks))

    def read_number(self):
        self.update_current_numbers()
        self.update_plot()
        print(self._make_output())
        read_string(str(self.current_numbers[1]))

    def load_data(self, skip_unnecessary=True):
        self.data = np.genfromtxt(self.data_path, dtype=int, delimiter=',')
        self.data[:, 1:] = (self.data[:, 1:] * -1 - 125) % 500
        self.all_hooks = self.data[:, 2]
        if skip_unnecessary:
            not_skip_hooks = ~np.asarray(get_skip_hooks(self.all_hooks))
            self.data = self.data[not_skip_hooks, :]
            self.data[:, 0] = range(len(self.data))
            self.all_hooks = self.all_hooks[not_skip_hooks]

        self.all_hooks = self.all_hooks.tolist()
        self.update_current_numbers()

    def plot_overview(self, fig=None, ax=None, color='#AC9F3C', setup=False):
        xlist = []
        ylist = []
        if (fig is None) or (ax is None):
            fig, ax = plt.subplots()
            setup = True
        self.fig = fig
        self.ax = ax

        self.connect()

        length_string = 0
        string_order = self.data[:, 1:]
        shape = self.rectangle
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

        for h in range(self.counter):
            ax.plot([shape[string_order[h][0], 0], shape[string_order[h][1], 0]],
                    [shape[string_order[h][0], 1], shape[string_order[h][1], 1]], 'b', linewidth=.1)

        for h in range(0, 500, 25):
            plt.scatter(shape[h, 0], shape[h, 1], color='c')
            plt.text(shape[h, 0], shape[h, 1], h)

        self.status_plot = self.ax.scatter(shape[string_order[self.counter][1], 0],
                                           shape[string_order[self.counter][1], 1], color='r')

        if setup:
            ax.set_aspect('equal')
            ax.set_axis_off()
        return fig, ax


def main(file, restart=False, compute_dir='.'):
    hook_reader = HookReader(file, overwrite=restart, compute_dir=compute_dir)
    fig, ax = plt.subplots()
    hook_reader.plot_overview(fig=fig, ax=ax, setup=True)


    read_string('starting hook reader')
    stop_recognizer = hook_reader.start()
    plt.show()
    print('stop recognizer')
    stop_recognizer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', type=str, help='hook order file')
    parser.add_argument('-t', '--tmp', default='.', type=str, help='temporary file dir')
    parser.add_argument('-r', '--restart', action='store_true', help='whether to restart')

    args = parser.parse_args()
    main(args.file, args.restart, args.tmp)
