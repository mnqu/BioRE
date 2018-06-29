import sys
import os
import struct

def read_emb (file_name):
	data = {}
	fi = open(file_name, 'rb')
	line = fi.readline()
	size = int(line.split()[0])
	dims = int(line.split()[1])
	print size, dims
	for i in range(size):
		pst = fi.tell()
		cnt = 0
		while True:
			ch = fi.read(1)
			cnt += 1
			if ch == ' ':
				break
		fi.seek(pst)
		name = fi.read(cnt).strip()
		emb = []
		for j in range(dims):
			f = struct.unpack('f',fi.read(4))
			emb.append(f[0])
		fi.read(1)
		data[name] = emb
	return data 

def write_emb (file_name, data):
	fo = open(file_name, 'wb')
	size = len(data)
	dims = len(data.values()[0])
	print size, dims
	fo.write(str(size) + ' ' + str(dims) + '\n')
	for name, emb in data.items():
		fo.write(name + ' ')
		for f in emb:
			fo.write(struct.pack('f', f))
		fo.write('\n')
	fo.close()

if __name__ == '__main__':
	data = read_emb(sys.argv[1].strip())
	write_emb(sys.argv[2].strip(), data)
	