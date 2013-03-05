CC=gcc
LIBS=-lm -lrt -lmongoc

all: src/khann.c src/main.c
	$(CC) -o khann src/khann.c src/main.c -std=c99 $(LIBS)
	python setup.py build
	mv build/lib.linux-x86_64-2.7/khann.so api
	rm -r build
