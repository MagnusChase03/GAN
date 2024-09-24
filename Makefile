SRC = $(shell find src | grep -E '.*\.cpp')
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))

build: $(OBJ)
	mkdir -p $(@D)
	g++ -shared -o build/gan.so $^
	strip build/gan.so

build/%.o: src/%.cpp
	mkdir -p $(@D)
	g++ -c -fPIC -o $@ $^

clean:
	rm -r build
