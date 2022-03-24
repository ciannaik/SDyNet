import numpy as np
import warnings


class MaximumIterationsExceededError(Exception):
    """ Error raised when iterations of a loop exceeds a predefined limit. """
    pass


def linear_slice_step(x_curr, log_f_curr, log_f_func, *args, slice_width, prng,
                      max_steps_out=0, max_slice_iters=1000):
    """ Performs single linear slice sampling update.
    Performs slice sampling along some line in the target distribution state
    space. This line might be axis-aligned corresponding to sampling along
    only one of the dimensions of the target distribution or some arbitrary
    linear combination of the dimensions.
    The first step in a slice sampling update is to randomly sample a slice
    height between 0 and the (potentially unnormalised) density at the current
    Markov chain state. The set of all the points on the line with a density
    above this slice height value are defined as the current slice and moving
    to a state corresponding to a point drawn uniformly from this slice on
    the line will leave the target distribution invariant. To achieve this
    the first step is to randomly position a bracket of a specified width
    around the current point on the line, optionally stepping out this bracket
    until its ends lie outside the slice. Points are then repeatedly drawn at
    uniform from the current bracket and tested to see if they are in the slice
    (i.e. have a density above the slice height), if they are the current point
    returned otherwise rejected proposed points are used to adaptively shrink
    the slice bracket while maintaining the reversibility of the algorithm.
    **Reference:**
    `Slice sampling`, Neal (2003)
    Parameters
    ----------
    x_curr : ndarray
        Point on line corresponding to current Markov chain state.
    log_f_curr : float
        Logarithm of the potentially unnormalised target density evaluated at
        current state.
    log_f_func : function or callable object
        Function which calculates the logarithm of the potentially unnormalised
        target density at a point on the line. Should have call signature::
            log_f = log_f_func(x)
        where ``x`` is the position on the line to evaluate the density at and
        ``log_f`` the calculated log density.
    slice_width : float
        Initial slice bracket width with bracket of this width being randomly
        positioned around current point.
    prng : RandomState
        Pseudo-random number generator object (either an instance of a
        ``numpy`` ``RandomState`` or an object with an equivalent
        interface).
    max_steps_out : integer
        Maximum number of stepping out iterations to perform (default 0). If
        non-zero then the initial slice bracket  is linearly 'stepped-out' in
        positive and negative directions by ``slice_width`` each time, until
        either the slice bracket ends are outside the slice or the maximum
        number of steps has been reached.
    max_slice_iters : integer
        Maximum number of slice bracket shrinking iterations to perform
        before terminating and raising an ``MaximumIterationsExceededError``
        exception. This should be set to a relatively large value (e.g. the
        default is 1000) which is significantly larger than the expected number
        of slice shrinking iterations so that this exception is only raised
        when there is some error condition e.g. when there is a bug in the
        implementation of the ``log_f_func`` which would otherwise cause the
        shriking loop to never be terminated.
    Returns
    -------
    x_next : ndarray
        Point on line corresponding to new Markov chain state after performing
        update - if previous state was distributed according to target density
        this state will be too.
    log_f_next : float
        Logarithm of target density at updated state.
    Raises
    ------
    MaximumIterationsExceededError
        Raised when slice shrinking loop does not terminate within the
        specified limit.
    """
    #
    # The MIT License (MIT)
    #
    # Copyright (c) 2015 Matt Graham
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.

    # draw random log slice height between -infinity and log_f_curr
    log_y = np.log(prng.uniform()) + log_f_curr
    # randomly set initial slice bracket of specified width w
    x_min = x_curr - slice_width * prng.uniform()
    x_max = x_min + slice_width
    # step out bracket if non-zero maximum steps out
    if max_steps_out > 0:
        # randomly split maximum number of steps between up and down steps
        # to ensure reversibility
        steps_down = np.round(prng.uniform() * max_steps_out)
        steps_up = max_steps_out - steps_down
        s = 0
        while s < steps_down and log_y < log_f_func(x_min,*args):
            x_min -= slice_width
            s += 1
        s = 0
        while s < steps_up and log_y < log_f_func(x_max,*args):
            x_max += slice_width
            s += 1
    i = 0
    while i < max_slice_iters:
        # draw new proposed point randomly on current slice bracket and
        # calculate log density at proposed point
        x_prop = x_min + (x_max - x_min) * prng.uniform()
        log_f_prop = log_f_func(x_prop,*args)
        # check if proposed state on slice if not shrink
        if log_f_prop > log_y:
            return x_prop, log_f_prop
        elif x_prop < x_curr:
            x_min = x_prop
        elif x_prop > x_curr:
            x_max = x_prop
        else:
            warnings.warn('Slice collapsed to current value')
            return x_curr, log_f_curr
        i += 1
    raise MaximumIterationsExceededError(
        'Exceed maximum slice iterations: '
        'i={0}, x_min={1}, x_max={2}, log_f_prop={3}, log_f_curr={4}'
        .format(i, x_min, x_max, log_f_prop, log_f_curr))