using System;

namespace NeuralNetworkExample01
{

    public class MatrixNn
    {
        public double[,] Values { get; }

        public int Rows { get => Values.GetLength(0); }
        public int Columns { get => Values.GetLength(1); }

        public MatrixNn(int rows, int columns) 
        {
            Values = new double[rows, columns];
        }

        public MatrixNn(double[,] values)
        {
            Values = values;
        }

        public MatrixNn(double[] vector)
        {
            /// Creating Matrix with X rows and 1 column
            Values = new double[vector.Length, 1];
            for (int i = 0; i < vector.Length; ++i)
            {
                Values[i, 0] = vector[i];
            }
        }

        public MatrixNn Transpose()
        {
            var newMatrix = new MatrixNn(Columns, Rows);
            for (int i = 0; i < Rows; ++i)
            {
                for (int j = 0; j < Columns; ++j)
                {
                    newMatrix.Values[j, i] = Values[i, j];
                }
            }
            return newMatrix;
        }

        public MatrixNn ApplyFunction(Func<double, double> valueTransform)
        {
            var newMatrix = new MatrixNn(Rows, Columns);
            for (int i = 0; i < Rows; ++i)
            {
                for (int j = 0; j < Columns; ++j)
                {
                    newMatrix.Values[i, j] = valueTransform(Values[i, j]);
                }
            }
            return newMatrix;
        }

        public MatrixNn SubstractMatrixes(MatrixNn other)
        {
            if (Columns != other.Columns || Rows != other.Rows)
                throw new ArgumentException("Matrixes can't be substracted!");
            var result = new MatrixNn(Rows, Columns);

            for (int i = 0; i < Rows; ++i)
            {
                for (int j = 0; j < Columns; ++j)
                {
                    result.Values[i, j] = Values[i, j] - other.Values[i, j];
                }
            }

            return result;
        }

        public MatrixNn MultiplyByMatrix(MatrixNn other)
        {
            var A = Values;
            int rA = A.GetLength(0);
            int cA = A.GetLength(1);
            if (rA == 1 && cA == 1)
            {
                var singleValue = Values[0,0];
                var result = other.ApplyFunction((e) => e * singleValue);
                return result;
            }

            var B = other.Values;
            int rB = B.GetLength(0);
            int cB = B.GetLength(1);

            if (cA != rB)
            {
                Console.WriteLine("> Can not multiply matrix A:");
                PrintMatrix();
                Console.WriteLine("> by matrix B:");
                other.PrintMatrix();
                throw new ArgumentException("Matrixes can't be multiplied!!");
            }
            else
            {
                double temp = 0;
                double[,] kHasil = new double[rA, cB];

                for (int i = 0; i < rA; i++)
                {
                    for (int j = 0; j < cB; j++)
                    {
                        temp = 0;
                        for (int k = 0; k < cA; k++)
                        {
                            temp += A[i, k] * B[k, j];
                        }
                        kHasil[i, j] = temp;
                    }
                }

                return new MatrixNn(kHasil);
            }
        }
        public void PrintMatrix()
        {
            var matrix = Values;
            Console.WriteLine($"--- Matrix (columns={Columns}, Rows={Rows}) print: ---");
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    Console.Write(matrix[i, j] + ";");
                }
                Console.WriteLine();
            }
            Console.WriteLine("--- Matrix print end ---");
        }
    }

}
