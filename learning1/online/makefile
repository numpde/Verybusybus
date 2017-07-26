default:
	cat makefile

clean:
	rm -f *.eps *.png *.dat *.pkl *.log NN1.h5 
	python3 aux_*.py

run:
	python3 online.py 1>  cout.log 2>  cerr.log
	python3 online.py 1>> cout.log 2>> cerr.log
