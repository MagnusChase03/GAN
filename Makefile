SRC = $(shell find src | grep -E '*.cpp')
OBJ = $(patsubst src/%.cpp,build/%.o,$(SRC))

build: $(OBJ)
	mkdir -p $(@D)
	g++ -o build/main $^

build/%.o: src/%.cpp
	mkdir -p $(@D)
	g++ -c -o $@ $^

clean:
	rm -r build
