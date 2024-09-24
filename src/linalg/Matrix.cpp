#include "Matrix.h"
#include <stdexcept>
#include <cstdio>

Matrix::Matrix(int rows, int cols) {
    if (rows < 1 || cols < 1) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    this->rows = rows;
    this->cols = cols;

    this->values = std::vector<std::vector<double>>(rows);
    for (int i = 0; i < rows; i++) {
        this->values[i] = std::vector<double>(cols);
    }
}

Matrix::Matrix(std::vector<std::vector<double>> values) {
    this->rows = values.size();
    this->cols = values[0].size();

    this->values = std::vector<std::vector<double>>(this->rows);
    for (int i = 0; i < this->rows; i++) {
        this->values[i] = std::vector<double>(this->cols);
        for (int j = 0; j < this->cols; j++) {
            this->values[i][j] = values[i][j];
        }
    }
}

double Matrix::getValue(int row, int col) {
    if (row < 0 || row >= this->rows || col < 0 || col >= this->cols) {
        throw std::runtime_error("[ERROR] Invalid index");
    }

    return this->values[row][col];
}

void Matrix::setValue(int row, int col, double value) {
    if (row < 0 || row >= this->rows || col < 0 || col >= this->cols) {
        throw std::runtime_error("[ERROR] Invalid index");
    }

    this->values[row][col] = value;
}


Matrix* Matrix::dot(Matrix* other) {
    if (this->cols != other->getRows()) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    Matrix* result = new Matrix(this->rows, other->getCols());
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            for (int k = 0; k < other->getCols(); k++) {
                double old = result->getValue(i, k);
                double product = this->values[i][j] * other->getValue(j, k);
                result->setValue(i, k, old + product);
            }
        }
    }

    return result;
}


Matrix* Matrix::add(Matrix* other) {
    if (this->rows != other->getRows() || this->cols != other->getCols()) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    Matrix* result = new Matrix(this->rows, this->cols);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            double sum = this->values[i][j] + other->getValue(i, j);
            result->setValue(i, j, sum);
        }
    }

    return result;
}

Matrix* Matrix::mult(double value) {
    Matrix* result = new Matrix(this->rows, this->cols);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            double product = this->values[i][j] * value;
            result->setValue(i, j, product);
        }
    }

    return result;
}

void Matrix::print() {
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            printf("%.2f ", this->values[i][j]);
        }
        printf("\n");
    }
}
