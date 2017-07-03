tls = open("passed", "r")
coup = open("cc_ec.parsed", "r")

out = open("tls.dat", "wb")

while True:
	line = tls.readline()
	if(not line):
		break
	outList =  line.split() + coup.readline().split()
	outString = "\t".join(outList)
	out.write(outString + "\n")
