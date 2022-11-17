import sys
import time
from IPython.display import clear_output
import datetime
import numpy as np
from typing import Any


class tqdm_jupyter:
    # This two class attributes as necessary for nested loops
    output = ''
    had_nested = 0

    def __init__(self, iterable, desc: str = ''):
        self.iterable = iterable
        self.iterator = iter(self.iterable)
        self.desc = desc
        self.index = 0
        self.start = time.time()
        self.start_next = time.time()
        self.output = tqdm_jupyter.output
        # If we had previous nested loops and are not on the highest level loop, we decrement
        # the had_nested attribute, since this was increased at the end of the lower level loop
        #  which would then produce a too large output field.
        if self.output and tqdm_jupyter.had_nested > 0:
            tqdm_jupyter.had_nested -= 1
        else:
            tqdm_jupyter.had_nested = 0

    def __iter__(self):
        """This method will be called at instanciation of the loop. It has to return the object"""
        return self

    def __len__(self):
        """This method returns the length of the iterable if len(tqdm_jupyter(iterable)) is called"""
        return len(self.iterable)

    def calculate_time_delta(self) -> None:
        """This method calculates the time that has passed since the loop has started."""
        if self.index == 0:
            self.time_delta = 0
        else:
            self.time_delta = round(time.time() - self.start)

    def calculate_its(self) -> None:
        """This method calculates how many iterations could be done in a second
        extrapolating from the last iteration of the loop."""
        if self.index == 0:
            self.its = '  ?'
        else:
            self.its = format(1 / (time.time() - self.start_next), ".2f")

    def set_description(self, desc: str = '') -> None:
        """This method prepends a custom description to the progress bar."""
        self.desc = desc
        # Since at the end of the __next__ method, we increase the index, we will have to decrement
        # it here first before we can call the get_constant_string method again which depends on the
        #  index.
        self.index -= 1
        # Producing the progress message
        progress = self.output + f'{self.desc} ' + self.get_constant_string()
        # Once we have our personalized message, we can increase the index again
        self.index += 1
        # We replace the old message with our new one
        sys.stderr.write(progress + '\n' * (tqdm_jupyter.had_nested))
        sys.stderr.flush()
        clear_output(wait=True)
        # We assign our new message to the class attribute such that nested loops will be able to
        # access it.
        tqdm_jupyter.output = progress

    def get_constant_string(self) -> str:
        """This method creates the basic progress bar string."""
        # Since the percentages in the progress bar vary between 1 and three digits, we have to adjust
        # for it with spaces to keep the rest of the progress bar in the same position. Here we
        # determine this space padding.
        if self.index == 0:
            add_space = '  '
        elif self.index == len(self.iterable):
            add_space = ''
        else:
            add_space = ' '
        # This defines how much of the actual progress bar is white (spaces) (the rest will be
        # colored). The progress bar should have a maximum length of 80 to make it fit on a
        #  normal sized jupyter window.
        spaces = 80 - len(f'{self.desc} ' + add_space +
                          f'{round((self.index) / len(self.iterable) * 100)}%|')
        # This defines the constant part of the progress message: padding, percentage, colored
        # part of progress bar, white part of progress bar, current iteration out of total
        #  iterations, time passed since loop instanciation, hypothetical iterations per second.
        constant_string = (add_space +
                           f'{round(self.index / len(self.iterable) * 100)}%|'
                           f'{"█" * int(np.floor((self.index / len(self.iterable)) * spaces))}'
                           f'{" " * int(np.ceil((1 - (self.index / len(self.iterable))) * spaces))}|'
                           f' {self.index}/{len(self.iterable)}'
                           f' [{datetime.timedelta(seconds=self.time_delta)}, '
                           f'{self.its}it/s]\n')
        return constant_string

    def __next__(self) -> Any:
        """This method will be called at each iteration. In here we print out the progress bar."""
        # If an empty iterable was passed, stop immediately. The same applies to if a finished
        # iterator is restarted.
        if len(self.iterable) == 0 or self.index > len(self.iterable):
            raise StopIteration
        # Calculate how long it has been since the loop was started
        self.calculate_time_delta()
        # Based on the last iteration, calculate how many iterations could be done in one second
        self.calculate_its()
        # Create the message to print out
        progress = self.output + f'{self.desc} ' + self.get_constant_string()
        # Print out the message. The extra new lines are because we want to keep the maximum
        # output size for nested loops
        sys.stderr.write(progress + '\n' * (tqdm_jupyter.had_nested))
        # We are flushing the buffer to the output (abundance of caution)
        sys.stderr.flush()
        # If something else will be printed, remove everything you have printed so far. This way
        # the ouput doesn't expand for every iteration (which would become huge considering the)
        # number of files we have to download.
        clear_output(wait=True)
        # Assign the output message to the class attribute output, such that nested loops can
        # access and print it before their own progress (since we are flushing the previous)
        # output.
        tqdm_jupyter.output = progress
        # If we iterated over everything and are in the next iteration in the loop (one more)
        # than there are elements in the iterable, we adjust the parameters had_nested which
        # defines if there were nested loops (this information is required by higher level
        # lopps), we also adjust the output attribute if we are on the highest level loop.
        # Finally we stop the iteration.
        if self.index == len(self.iterable):
            if self.output:
                tqdm_jupyter.had_nested += 1
            else:
                tqdm_jupyter.had_nested = 0
                tqdm_jupyter.output = ''
            raise StopIteration
        # We increase the attribute that keeps track on where we are in the loop
        self.index += 1
        # We start the timer to measure the execution
        self.start_next = time.time()
        # Finally we return the next element of the iterator
        return next(self.iterator)
