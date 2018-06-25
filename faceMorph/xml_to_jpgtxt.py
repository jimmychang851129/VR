# XML from https://naturalintelligence.github.io/imglab/
filename = "labelled.xml"
with open(filename, "r") as fp:
	with open("labelled.jpg.txt","w") as fr:
		while 1:
			line = fp.readline()
			if line == "":
				break
			if line.find("x=") != -1:
				ll = line.split("=\'")
				num1 = ll[2][0:ll[2].find(".")] 
				num2 = ll[3][0:ll[3].find(".")]
				fr.write(num1+" "+num2+"\n")