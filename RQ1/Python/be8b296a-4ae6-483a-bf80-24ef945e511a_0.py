import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    print("Device {} - ID: {} Name: {}".format(i, device_info['index'], device_info['name']))

p.terminate()
