"""
Copyright (C) 2019 SensorLab, Jozef Stefan Institute http://sensorlab.ijs.si

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses
"""

from __future__ import print_function

from tempfile import mkdtemp
import os
import sys
import signal

from collections import defaultdict
from datetime import datetime
import random
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Cursor
from matplotlib import cm
from matplotlib import ticker

from PIL import Image


# Parameters.
base_data_path = "./"
spectrum_f_names = ["ws_traffic_20170606"]
spectrum_data_files = [os.path.join(base_data_path, x) for x in spectrum_f_names]

output_directory = "./"
window_duration = 30 # seconds
rand_skip_forward_range = (10*60, 15*60) # seconds
noise_cutoff = None # Set spectrogram to minimum value where value <= minimum value + noise_cutoff [dBm].

random.seed(42)


def signal_handler(signal, frame):
    print("Exiting...")
    sys.exit(0)


def generate_cir_image(data_array, image_name=None):
     """
     Generates image with all CIRs according to the range
     :param data_array: complete fitered data arrays
     :param image_name: name of output image
     :return:
     """
     cir = data_array
     img_array = np.zeros([cir.shape[0], cir.shape[1], 3])

     # transform to image array
     for i in range(cir.shape[0]):
         for j in range(cir.shape[1]):
             img_array[i][j] = np.array([cir[i][j], cir[i][j], cir[i][j]])
     # normalize to 1
     img_array = img_array / np.max(img_array)
     lum_img = img_array[:, :, 0]
     # apply range 0 to 255
     img_array2 = lum_img * 255
     # apply color map "nipy_spectral"
     img_array2 = cm.nipy_spectral(lum_img) * 255
     img_array2 = img_array2.astype('uint8')
     img = Image.fromarray(img_array2)
     return img
     #img.save(image_name)


class SpectrumLabeler:
    def __init__(self, spectrum_data_files, output_directory, window_duration,
                  rand_skip_forward_range, min_window_points=5):
        self.spectrum_data_files = spectrum_data_files
        self.output_directory = output_directory
        self.window_duration = window_duration
        self.rand_skip_forward_range = rand_skip_forward_range
        self.events = None
        self.windows = None
        self.min_window_points = min_window_points

        self.min_v = None
        self.max_v = None

        # Automatically maximizing the plots causes too much problems on various platforms.
        # plt.switch_backend('QT4Agg')
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

    def run(self):
        for spectrum_data_file in spectrum_data_files:
            print("New dataset: %s" % spectrum_data_file)
            self.events = []
            self.windows = []
            data_p, data_t = self.__load_data(spectrum_data_file)
            
            data_t_start = data_t[0]
            data_t_end = data_t[-1]
            window_t_start = data_t_start + self.__get_rand_offset()# + self.__get_rand_offset()
            
            while window_t_start < data_t_end - 2*self.window_duration:

                window_i_start, window_i_end = self.__get_window_indices(data_t, window_t_start)
                # print("True start: %s, found: %s" % (window_t_start, data_t[window_i_start]))
                # print("True end: %s, found: %s" % (window_t_start + self.window_duration, data_t[window_i_end]))
                # Skip if the window doesn't comply with our requirements.
                # if window_i_end - window_i_start < self.min_window_points:
                #     continue
                if abs(window_t_start - data_t[window_i_start]) > self.window_duration / 4 or \
                 abs(window_t_start + self.window_duration - data_t[window_i_end]) > self.window_duration / 4 or \
                 abs(data_t[window_i_start] - data_t[window_i_end]) < self.window_duration / 2:
                    window_t_start += self.__get_rand_offset()
                    continue

                self.windows.append((data_t[window_i_start], data_t[window_i_end]))
                self.__display_spectrogram_record_events(data_p, data_t, window_i_start, window_i_end)
                
                print("Left to label: %d s of data." % (data_t_end - window_t_start))
                window_t_start += self.window_duration + self.__get_rand_offset()


            self.__output_to_file(spectrum_data_file)

    def __get_rand_offset(self):
        return random.randint(*self.rand_skip_forward_range)

    def __load_data(self, f_name):
        print("Memory mapping the file (this might take some time) ...")
        t = []
        with open("%s" % f_name, 'r') as f:
            num_lines = 1
            first_line = f.readline()
            j_line = json.loads(first_line)
            num_measurements = len(j_line["Measurements"])
            for _ in f:
                num_lines += 1

            f.seek(0)
            mmap_data = np.memmap(os.path.join(mkdtemp(), 'tmpspec.bin'),
                                  mode='w+', shape=(num_lines, num_measurements), dtype=np.float64)
            for i, line in enumerate(f):
                j_line = json.loads(line)
                ts = datetime.strptime(j_line["Time"], '%Y-%m-%dT%H:%M:%S.%f').timestamp()
                t.append(ts)
                mmap_data[i,:] = j_line["Measurements"]

            #mmap_data.flush()
            print("Done!")
        self.max_v = np.max(mmap_data)
        self.min_v = np.min(mmap_data)

        return mmap_data, t

    def __get_window_indices(self, t_data, window_t_start):
        start = np.searchsorted(t_data, window_t_start, side='right')
        end = np.searchsorted(t_data, window_t_start + self.window_duration) - 1
        return start, end

    def __display_spectrogram_record_events(self, data_p, data_t, window_i_start, window_i_end):
        # Show the spectrogram and record input.
        fig, ax = plt.subplots(figsize=(14, 7))

        part_p = data_p[window_i_start : window_i_end].copy()
        if noise_cutoff:
            part_p[part_p <= self.min_v + noise_cutoff] = self.min_v
        scaled_p = (part_p - self.min_v) / (self.max_v - self.min_v)

        img = ax.imshow(scaled_p, interpolation='none',
                   origin='lower', aspect='auto', cmap='inferno')
        plt.xlabel("FFT bin")
        plt.ylabel("Time (each point is approx. %.3f s)" % ((data_t[window_i_end] - data_t[window_i_start]) \
                                                              / (window_i_end - window_i_start)) )
        cbar = plt.colorbar(img)

        img.set_clim(vmin=0, vmax=1)
        cbar.set_ticks([1, 0.5, 0])
        #tick_locator = ticker.MaxNLocator(nbins=3)
        #cbar.locator = tick_locator
        #cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())

        #cbar.update_ticks()
        cbar.ax.set_yticklabels([self.max_v, (self.max_v + self.min_v) / 2, self.min_v])

        with EvaluationInputRecorder(fig, ax, window_i_start) as evaluation_recorder:
            cursor = Cursor(ax, useblit=True, color='white', linewidth=1)
            plt.show(block=False)
            try:
                mouse = plt.waitforbuttonpress()
                while not mouse:
                    mouse = plt.waitforbuttonpress()
            except KeyboardInterrupt:
                sys.exit(0)
            #raw_input("< press ENTER when done >\n")
            plt.close()

            self.events.append(map(lambda x: dict(x, **{"StartTime": data_t[x["StartTime"]], "EndTime": data_t[x["EndTime"]]}),
                                   evaluation_recorder.tx_events))

    def __output_to_file(self, f_name):
        with open(self.output_directory + "out_" + f_name.split("/")[-1], 'w') as f:
            for window, events in zip(self.windows, self.events):
                f.write("%f %f\n" % (window[0], window[1]))
                for event in events:
                    f.write("%s\n" % event)
                f.write("\n")



class EvaluationInputRecorder:
    def __init__(self, fig, ax, offset=0):
        self.fig = fig
        self.offset = offset
        self.cid_p = fig.canvas.mpl_connect('button_press_event', self)
        self.cid_r = fig.canvas.mpl_connect('button_release_event', self)
        self.tx_events = []
        self.display_marks = []
        self.display_rects = []
        self.ax = ax
        self.last_e = None

    def __call__(self, event):
        tb = plt.get_current_fig_manager().toolbar
        if event.inaxes and tb.mode == '':
            event.ydata = int(round(event.ydata))
            event.xdata = int(round(event.xdata))
            if event.name == "button_press_event":
                if event.button == 1:
                    print("Time: %s, FFT bin: %s" % (event.ydata, event.xdata))
                    mark, = self.ax.plot(event.xdata, event.ydata, marker='o', color='white')
                    self.display_marks.append(mark)
                    self.fig.canvas.draw()
                    # self.tx_events.append((event.ydata + self.offset, event.xdata, "start"))
                    # print(" Transmisson start (left edge).")
                    self.last_e = {"xdata": event.xdata, "ydata": event.ydata}
                elif event.button == 3:
                    if len(self.tx_events) > 0:
                        self.display_marks.pop().remove()
                        self.display_marks.pop().remove()
                        self.display_rects.pop().remove()
                        ev = self.tx_events.pop()
                        print("Removing the last event: %s" % ev)
                    else:
                        print("Can't remove: no events.")
            elif event.name == "button_release_event" and event.button == 1 and self.last_e:
                # print("Time: %s, FFT bin: %s" % (event.ydata, event.xdata), end="")
                # mark, = self.ax.plot(event.xdata, event.ydata, marker='o', color='white')
                # self.display_marks.append(mark)
                # self.fig.canvas.draw()
                # self.tx_events.append((event.ydata + self.offset, event.xdata, "stop"))
                # print(" Transmisson stop (right edge).")
                
                print("Time: %s, FFT bin: %s" % (event.ydata, event.xdata))
                mark, = self.ax.plot(event.xdata, event.ydata, marker='o', color='white')
                self.display_marks.append(mark)

                rect = patches.Rectangle((self.last_e["xdata"], self.last_e["ydata"]),
                                         event.xdata - self.last_e["xdata"],
                                         event.ydata - self.last_e["ydata"],
                                         linewidth=1, edgecolor='w', fill=False)
                self.display_rects.append(rect)
                self.ax.add_patch(rect)

                self.fig.canvas.draw()

                # Add offset.
                self.tx_events.append({"StartChannel": min(event.xdata, self.last_e["xdata"]),
                                       "EndChannel": max(event.xdata, self.last_e["xdata"]),
                                       "StartTime": min(event.ydata, self.last_e["ydata"]) + self.offset,
                                       "EndTime": max(event.ydata, self.last_e["ydata"]) + self.offset})

                self.last_e = None

            # else:
            #     print("Unknown event.")
            #     self.last_e = None
        else:
            print("Please click inside the spectrogram.")
            if self.last_e:
                self.display_marks.pop().remove()
                self.fig.canvas.draw()
            self.last_e = None

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.fig.canvas.mpl_disconnect(self.cid_p)
        self.fig.canvas.mpl_disconnect(self.cid_r)





if __name__ == "__main__":
    # Gracefully exit the script.
    signal.signal(signal.SIGINT, signal_handler)

    print("Label the transmissions: left click, hold and release to draw a rectangle around a single transmission."
            " The order or orientation of labels does not matter. To undo, press the right mouse button."
            " Press <space> (or almost any other key really) to display the next spectrogram.")

    labeler = SpectrumLabeler(spectrum_data_files, output_directory, window_duration,
                               rand_skip_forward_range)
    labeler.run()