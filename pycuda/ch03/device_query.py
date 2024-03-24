import pycuda.driver as drv

drv.init()

print('Detected {} CUDA Capable Device(s)'.format(drv.Device.count()))

for i in range(drv.Device.count()):
  gpu_device = drv.Device(i)
  print('Device {} {}'.format(i, gpu_device.name()))
  print('\t Compute capability: {}'.format(gpu_device.compute_capability()))
  print('\t Total Memory: {} GB'.format(gpu_device.total_memory()/(1024**3)))
  
  device_attr = gpu_device.get_attributes()
  for v in device_attr:
    print('{}\t{}'.format(v, gpu_device.get_attribute(v)))
  # print('MULTIPROCESSOR_COUNT: '.format(device_attr[drv.device_attribute.MULTIPROCESSOR_COUNT]))

  