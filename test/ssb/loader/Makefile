loader: load_modified.c hyper_loader.cpp hyper_read.cpp multinomial_test.cpp
	gcc -o loader load_modified.c
	g++ -o multinomial_test `gsl-config --cflags` multinomial_test.cpp `gsl-config --libs`
	g++ -o hyperLoader -isystem /home/zhaoh/crystal/hyper/include -Wl,-rpath,/home/zhaoh/crystal/hyper/lib hyper_loader.cpp /home/zhaoh/crystal/hyper/lib/libtableauhyperapi.so
	g++ -o hyper_read -isystem /home/zhaoh/crystal/hyper/include -Wl,-rpath,/home/zhaoh/crystal/hyper/lib hyper_read.cpp /home/zhaoh/crystal/hyper/lib/libtableauhyperapi.so

original_loader: load.c
	gcc -o gpuDBLoader load.c

sort: columnSort.c
	gcc -o columnSort columnSort.c -std=c99 

rle: rle.c
	gcc -std=c99 rle.c -o rleCompression

dict: dict.c
	gcc -std=c99 dict.c -o dictCompression

clean:
	rm -rf *.o gpuDBLoader columnSort rleCompression dictCompression loader hyperLoader hyper_read
