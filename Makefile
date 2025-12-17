CC=mpicc
#CC=mpiicx
CFLAGS=-O3 -march=native -flto -funroll-loops -fopenmp
#CFLAGS=-O3 -fopenmp 
LDFLAGS=-lm 

all: main.x

obj=src/main/main.c src/tree/tree.c src/common/common.c 
main.x: ${obj} 
	${CC} ${CFLAGS} ${obj} ${LDFLAGS} -o $@

clean:
	rm main.x
