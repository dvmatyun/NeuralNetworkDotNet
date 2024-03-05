// See https://aka.ms/new-console-template for more information
using NeuralNetworkExample01;
using System.Text;

Console.WriteLine("Hello, World!");
/*
var matrix1 = new MatrixNn(2, 1);
for (int i = 0; i < 2; i++)
{
    matrix1.Values[i, 0] = i + 1;
}
matrix1.PrintMatrix();
Console.WriteLine("Matrix transpose:");
var transposed = matrix1.Transpose();
transposed.PrintMatrix();

Console.WriteLine("Matrix multiply:");


var matrix2 = new MatrixNn(1, 2);
for (int i = 0; i < 2; i++)
{
    matrix2.Values[0, i] = i * 2 + 1;
}
matrix2.PrintMatrix();

var multiplied = matrix1.MultiplyByMatrix(matrix2);
multiplied.PrintMatrix();
*/
//

/*
var nnPerceptron = new NnPerceptronSimple(3, 2, 4);
var features = new double[3] { 0.5, 0.2, 0.4};
var output = new double[2] { 1, 1 };
*/
var nnPerceptron = new NnPerceptronSimple(3, 2, 8);
/*
var features = new double[2] { 0.5, 0.0 };
var output = new double[1] { 1 };

var matrixFeatures = new MatrixNn(features);
var matrixOutput = new MatrixNn(output);

// Write test file:
var file = new System.IO.StreamWriter("features.csv");
for (int i =0; i < 4; i++)
{
    file.WriteLine(nnPerceptron.GenerateExampleString());
}
file.Close();
*/
// Read file:

var testInputes = new List<TestInput>();
const Int32 BufferSize = 128;
using (var fileStream = File.OpenRead("features_read.csv"))
using (var streamReader = new StreamReader(fileStream, Encoding.UTF8, true, BufferSize))
{
    String line;
    while ((line = streamReader.ReadLine()) != null)
    {
        // Process line
        var input = TestInput.ParseInput(line, 3, 2);
        testInputes.Add(input);
    }
}

//Console.WriteLine("Matrix features:");
//matrixFeatures.PrintMatrix();

for (int i = 0; i < 100; i++)
{
    foreach (var input in testInputes)
    {
        var result = nnPerceptron.ForwardAndBackwardPropagation(input.Input, input.Output);
        if (i % 10 == 0)
        {
            Console.WriteLine($"> Error of prediction for #{i} (input={input.RawLine}): {String.Format("{0:0.000}", result.SumError)}");
        }
        
    }
}

var inputFinal = TestInput.ParseInput("0,2;0,2;0,4;1;0", 3, 2);
var resultFinal = nnPerceptron.ForwardPropagation(inputFinal.Input);
Console.WriteLine($"> Final prediction:");
resultFinal.LayerResults[1].SigmoidApplied.PrintMatrix();

//var result = nnPerceptron.ForwardPropagation(matrixFeatures);
/*
for (int i = 0; i < 100; ++i)
{
    var result = nnPerceptron.ForwardAndBackwardPropagation(matrixFeatures, matrixOutput);
    Console.WriteLine($"> Error of prediction: {result.SumError}");
}
*/


//Console.WriteLine($"Result of propagation:");
//result.ResultMatrix?.PrintMatrix();

public class TestInput
{
    public MatrixNn Input { get; }
    public MatrixNn Output { get; }

    public string RawLine { get; set; } = string.Empty;

    public TestInput(MatrixNn input, MatrixNn output)
    {
        Input = input;
        Output = output;
    }

    public static TestInput ParseInput(string line, int features, int outputs)
    {
        var splitted = line.Split(';').Select(double.Parse).ToArray();
        var featuresRead = splitted.Take(features).ToArray();
        var outputRead = splitted.TakeLast(outputs).ToArray();
        var input = new TestInput(new MatrixNn(featuresRead), new MatrixNn(outputRead)) { RawLine = line };
        return input;
    }
}