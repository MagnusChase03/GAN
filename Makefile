SRC = $(shell find src | grep -E '.*\.cpp')
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))

EX_SRC = $(shell find example | grep -E '.*\.cpp')
EX_OBJ = $(patsubst example/%.cpp,build/example/%.o,$(EX_SRC))

build/libgan.so: $(OBJ)
	mkdir -p $(@D)
	g++ -shared -o $@ $^
	strip build/gan.so

build/%.o: src/%.cpp
	mkdir -p $(@D)
	g++ -c -fPIC -o $@ $^

example: build/example/main

build/example/main: build/libgan.so $(EX_OBJ)
	mkdir -p $(@D)
	g++ -o $@ $(EX_OBJ) -Lbuild -lgan -Isrc

build/example/%.o: example/%.cpp
	mkdir -p $(@D)
	g++ -c -o $@ $^ -Isrc

run: build/libgan.so build/example/main
	LD_LIBRARY_PATH=build:$$LD_LIBRARY_PATH ./build/example/main

clean:
	rm -r build
