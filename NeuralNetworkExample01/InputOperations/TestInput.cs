using NeuralNetworkExample01.NeuralNetworkRaw;
using System.Text;

namespace NeuralNetworkExample01.InputOperations
{
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

        public string ToCsvString()
        {
            var features = Input.Values.GetLength(0);
            var outputs = Output.Values.GetLength(0);

            var sb = new StringBuilder();
            for (int i = 0; i < features; ++i)
            {
                sb.Append($"{Input.Values[i, 0]};");
            }
            for (int i = 0; i < outputs; ++i)
            {
                sb.Append($"{Output.Values[i, 0]};");
            }
            return sb.ToString();
        }
    }
}
