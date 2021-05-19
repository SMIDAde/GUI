"""
This function is stored in https://gitlab.com/GitPrinz/bokeh_scripts
"""
import time
from functools import partial

from bokeh.plotting import curdoc

##################
# delay function #
##################

# entries with function, callback_id, last_call
delay_keys = []
delay_callbacks = []
delay_last_call = []


def run_delayed(delay_ms, key, function, *arguments, verbose=False):
    """ Waiting some time if more call will happen and if not, then execute given function.

    :param delay_ms: time to wait
    :param key: unique key to track calls
    :param function: function to call
    :param n arguments: will be passed to function
    :param verbose: Will print what is happening.

    """
    global delay_callbacks, delay_keys, delay_last_call

    if verbose:
        print(str(time.time()) + " run_delayed(" + str(delay_ms) + "ms, " + key + ", " + str(function)
              + " with " + str(len(arguments)) + " Arguments)  Status["
              + str(len(delay_keys)) + "-" + str(len(delay_callbacks)) + "-" + str(len(delay_last_call)) + "]")

    if key in delay_keys:
        delay_id = delay_keys.index(key)

        if verbose:
            print(" Found match at id=" + str(delay_id)+". Last call "
                  + str((time.time() - delay_last_call[delay_id]))+" seconds ago.")

        if (time.time() - delay_last_call[delay_id]) > delay_ms / 1000:
            # when last call is longer ago then delay_ms, call the desired function.

            # remove periodic callback
            curdoc().remove_periodic_callback(delay_callbacks[delay_id])
            # case closed
            del delay_keys[delay_id]
            del delay_callbacks[delay_id]
            del delay_last_call[delay_id]

            if verbose:
                print(" Callback removed, running function.")

            function(*arguments)

        else:
            # when last call is not long ago, note the call and wait for next callback
            delay_last_call[delay_id] = time.time()

            if verbose:
                print(" Wait fore more calls.")

    else:
        # start periodic callback and note call

        if verbose:
            print(" Generating Callback.")

        delay_keys.append(key)
        delay_last_call.append(time.time())
        delay_callbacks.append(
            curdoc().add_periodic_callback(
                partial(run_delayed, delay_ms, key, function, *arguments,verbose=verbose),
                delay_ms))


# Testing
#   change to True and call "bokeh serve deayfun.py" via terminal
#   open the browser window
#   check the output in the command window.
# noinspection PyUnreachableCode
if False:
    print('>>start')
    ver = True

    run_delayed(500, "test", print, ">>This will be seen only once", verbose=ver)
    run_delayed(500, "test", print, ">>Second call will not be saved.", verbose=ver)
    run_delayed(1000, "test", print, ">>Only first call will go.", verbose=ver)
    run_delayed(2000, "test2", print, ">>Only if you change the key, something will happen again", verbose=ver)

    # bokeh is running every line here at once ... so we can't make an example with time.sleep(1)

