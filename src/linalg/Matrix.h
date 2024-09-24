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

    Matrix* dot(Matrix* other);
    Matrix* add(Matrix* other);
    Matrix* mult(double value);

    void print();
};

#endif
