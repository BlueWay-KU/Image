import socket
import os
import time

def transfer(filename):
    capture_file_name = str(filename) + ".jpg"
    file = open(capture_file_name, "rb")
    img_size = os.path.getsize(capture_file_name)
    img = file.read(img_size)
    file.close()

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("192.168.137.146", 49153))
    client_socket.sendall(img)
    client_socket.close()
    print("Done")

filename_list = [1]
for i in filename_list:
    transfer(i)
    time.sleep(3)