#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
private:
    std::vector<std::vector<double>> values;
    int rows;
    int cols;
public:
    Matrix(int rows, int cols);
    Matrix(std::vector<std::vector<double>> values);

    int getRows() {return rows;}
    int getCols() {return cols;}

    double getValue(int row, int col);
    void setValue(int row, int col, double value);

    Matrix* Dot(Matrix* other);
    Matrix* Add(Matrix* other);
    Matrix* Mult(double value);

    void Print();
};

#endif
