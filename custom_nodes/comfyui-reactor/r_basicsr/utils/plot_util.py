import re


def read_data_from_tensorboard(log_path, tag):
    """Get raw data (steps and values) from tensorboard events.

    Args:
        log_path (str): Path to the tensorboard log.
        tag (str): tag to be read.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # tensorboard event
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()
    scalar_list = event_acc.Tags()['scalars']
    print('tag list: ', scalar_list)
    steps = [int(s.step) for s in event_acc.Scalars(tag)]
    values = [s.value for s in event_acc.Scalars(tag)]
    return steps, values


def read_data_from_txt_2v(path, pattern, step_one=False):
    """Read data from txt with 2 returned values (usually [step, value]).

    Args:
        path (str): path to the txt file.
        pattern (str): re (regular expression) pattern.
        step_one (bool): add 1 to steps. Default: False.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    steps = []
    values = []

    pattern = re.compile(pattern)
    for line in lines:
        match = pattern.match(line)
        if match:
            steps.append(int(match.group(1)))
            values.append(float(match.group(2)))
    if step_one:
        steps = [v + 1 for v in steps]
    return steps, values


def read_data_from_txt_1v(path, pattern):
    """Read data from txt with 1 returned values.

    Args:
        path (str): path to the txt file.
        pattern (str): re (regular expression) pattern.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    data = []

    pattern = re.compile(pattern)
    for line in lines:
        match = pattern.match(line)
        if match:
            data.append(float(match.group(1)))
    return data


def smooth_data(values, smooth_weight):
    """ Smooth data using 1st-order IIR low-pass filter (what tensorflow does).

    Ref: https://github.com/tensorflow/tensorboard/blob/f801ebf1f9fbfe2baee1ddd65714d0bccc640fb1/\
        tensorboard/plugins/scalar/vz_line_chart/vz-line-chart.ts#L704

    Args:
        values (list): A list of values to be smoothed.
        smooth_weight (float): Smooth weight.
    """
    values_sm = []
    last_sm_value = values[0]
    for value in values:
        value_sm = last_sm_value * smooth_weight + (1 - smooth_weight) * value
        values_sm.append(value_sm)
        last_sm_value = value_sm
    return values_sm
