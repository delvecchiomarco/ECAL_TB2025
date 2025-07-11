try:
  import cupy as cp
except ImportError:
  import numpy as cp
try:
  from cupyx.scipy import ndimage
except ImportError:
  from scipy import ndimage

def get_reco_products(waves, signal_start, signal_end, rise_end, charge_thr, sampling_rate, nsamples, cf, save_waves):

  #waves has shape (nevents, chs, nsamples) - serie e parallelo separati
  pre_signal = waves[:, :, :int(signal_start*sampling_rate)] #da qua : a qua
  pre_signal_bline = pre_signal.mean(axis=2)
  pre_signal_rms = pre_signal.std(axis=2)

  waves = waves - cp.repeat(pre_signal_bline[:, :, cp.newaxis], nsamples, axis=2)

  #TGraph(...) -Draw().....
  #input()

  '''
  del pre_signal
  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  '''

  signal = waves[:, :, int(signal_start*sampling_rate):int(signal_end*sampling_rate)]

  if not save_waves:
    del waves

  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()

  charge = signal.sum(axis=2) / (50 * sampling_rate) #pC se parti da mV
  charge[charge<charge_thr] = 0

  charge_sum = charge.sum(axis=1)

  '''
  if charge.shape[1] == 18:
    _x = cp.asarray([int(i/2)%3-1 for i in range(18)])
    x = cp.repeat(_x[cp.newaxis, :], charge.shape[0], axis=0)
    _y = cp.asarray([int(int(i/2)/3)-1 for i in range(18)])
    y = cp.repeat(_y[cp.newaxis, :], charge.shape[0], axis=0)

    centroid_x = (x*charge).sum(axis=1)/charge_sum
    centroid_y = (y*charge).sum(axis=1)/charge_sum

  else:
    centroid_x = charge*0
    centroid_y = charge*0
  '''

  rise = signal[:, :, :int((rise_end-signal_start)*sampling_rate)]

  '''
  del signal
  cp.get_default_memory_pool().free_all_blocks()
  cp.get_default_pinned_memory_pool().free_all_blocks()
  '''

  #time_peak_not_interplated = rise.argmax(axis=2)/(sampling_rate*20)

  if rise.shape[0]==0:
    ampPeak = charge*0
    time_peak = charge*0
    pseudo_t = charge*0

  else:
    rise_interp = ndimage.zoom(rise, [1, 1, 20])

    '''
    del rise
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    '''

    ampPeak = rise_interp.max(axis=2)
    time_peak = rise_interp.argmax(axis=2)/(sampling_rate*20)

    pseudo_t = cp.argmax(rise_interp > cp.repeat((ampPeak*cf)[:, :, cp.newaxis], rise_interp.shape[2], axis=2), axis=2)/(sampling_rate*20)

  reco_dict = {
    "pre_signal_bline,": pre_signal_bline, "pre_signal_rms": pre_signal_rms,
    "charge": charge, "ampPeak": ampPeak, "time_peak": time_peak,
    "pseudo_t": pseudo_t, "charge_sum": charge_sum, "centroid_x": centroid_x, "centroid_y": centroid_y
  }

  if save_waves:  reco_dict.update({"wave": waves})
  return reco_dict
