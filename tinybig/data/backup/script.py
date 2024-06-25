
if 1:
	f = open('wheat', 'r')
	f_output = open('new_wheat', 'w')
	index = 0
	for line in f:
		line = line.strip('\n')
		if index == 0:
			f_output.write('"{}",\n'.format(line))
		else:
			elements = line.split()
			new_line = ','.join(elements)
			f_output.write('"{},{}",\n'.format(index, new_line))
		index += 1
	f.close()
	f_output.close()

if 0:
	f = open('banknote', 'r')
	f_output = open('new_banknote', 'w')
	index = 0
	for line in f:
		line = line.strip('\n')
		if index == 0:
			f_output.write('"{}",\n'.format(line))
		else:

			f_output.write('"{},{}",\n'.format(index, line))
		index += 1
	f.close()
	f_output.close()


if 0:
	f = open('pima_indians_diabetes.csv', 'r')
	f_output = open('new_pima_indians_diabetes.csv', 'w')
	index = 0
	for line in f:
		line = line.strip('\n')
		if index == 0:
			f_output.write('"{}",\n'.format(line))
		else:

			f_output.write('"{},{}",\n'.format(index, line))
		index += 1
	f.close()
	f_output.close()


# titanic
if 0:
	f = open('titanic.csv', 'r')
	f_output = open('new_titanic.csv', 'w')
	index = 0
	for line in f:
		line = line.strip('\n')
		line = line.replace(', ', '. ')
		line = line.replace('"', '')
		if index == 0:
			f_output.write('"{}",\n'.format(line))
		else:
			
			f_output.write('"{}{}",\n'.format(index,line))
		index += 1
	f.close()
	f_output.close()	

# wine
if 0:
	f = open('wine.csv', 'r')
	f_output = open('new_wine.csv', 'w')
	index = 0
	for line in f:
		line = line.strip('\n')
		if index == 0:
			f_output.write('"{}",\n'.format(line))
		else:
			
			f_output.write('"{}{}",\n'.format(index,line))
		index += 1
	f.close()
	f_output.close()	
	
