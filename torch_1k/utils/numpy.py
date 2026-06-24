from torch_1k import backend

def ensure_ndarray(data):
    return backend.ensure_array(data)


def np_sum_to(x, shape):
    '''
    Note:
        shape为目标shape
        根据目标shape确定是否keepdims
    '''
    if x.shape == shape:
        return x

    xp = backend.get_array_module(x)
    ndim = len(shape)
    lead = x.ndim - ndim
    if lead < 0:
        raise Exception(f'not-allowed: `np_sum_to` {x.shape} -> {shape}')

    lead_axis = tuple(range(lead))
    axis = tuple(i + lead for i, sx in enumerate(shape) if sx == 1)
    y = xp.sum(x, axis=lead_axis + axis, keepdims=True)
    if lead > 0:
        y = xp.squeeze(y, axis=lead_axis)
    return y
