using NeuralNetworkExample01.NeuralNetworkRaw;

namespace NeuralNetworkExample01
{
    public class NnConfig
    {
        public Dictionary<int, NnLayer> Layers { get; set; }

        public int Inputs { get; }
        public int Outputs { get; }

        public NnConfig(List<int> hiddenLayers, int inputAmount = 2, int outputAmount = 1) 
        {
            var inputLayer = new NnLayer(inputAmount, hiddenLayers.First());
            Layers = new Dictionary<int, NnLayer>
            {
                { 0, inputLayer },
                //{ 1, hiddenLayer }
            };

            for (int i = 0; i < hiddenLayers.Count; i++)
            {
                int nextAmount = i == (hiddenLayers.Count - 1) ? outputAmount : hiddenLayers[i + 1];
                var hiddenLayer = new NnLayer(hiddenLayers[i], nextAmount);
                Layers[i + 1] = hiddenLayer;
            }
            
            
            Inputs = inputAmount;
            Outputs = outputAmount;
        }

        #region Simple Functions
        public static double Sigmoid(double z)
        {
            var result = 1 / (1 + Math.Pow(double.E, -1 * z)); ;
            return result;
        }

        public static double SigmoidDerivative(double z)
        {
            var sigmoid = Sigmoid(z);
            return sigmoid * (1 - sigmoid);
        }
        #endregion
    }

    public class NnLayer
    {
        public MatrixNn Neurons { get; private set; }

        private MatrixNn? _neuronsTransposed;
        public MatrixNn NeuronsTransposed { get => _neuronsTransposed ??= Neurons.Transpose(); }

        public NnLayer(int rows, int columns)
        {
            Neurons = GetRandomMatrix(rows, columns);
        }

        public NnLayer(double[,] neurons)
        {
            Neurons = new MatrixNn(neurons);
        }

        private const int WeightGeneratorNum = 100;

        public void CorrectWeights(MatrixNn delta, double learningRate = 0.1)
        {
            var newDelta = delta.ApplyFunction((e) => e * learningRate);
            Neurons = Neurons.SubstractMatrixes(newDelta);
            _neuronsTransposed = null;
        }

        public static double GenerateRandomWeight(Random random)
        {
            return (double)random.Next(WeightGeneratorNum) / (double)WeightGeneratorNum;
        }
        public static MatrixNn GetRandomMatrix(int rows, int cols)
        {
            var rand = new Random();
            var arr = new MatrixNn(rows, cols);
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    arr.Values[i, j] = GenerateRandomWeight(rand);
                }
            }

            return arr;
        }
    }

    public class NnPropagationResult
    {
        public MatrixNn Features { get; set; } // X input
        public double Result { get; set; }
        public double SumError { get; set; }

        public MatrixNn? ResultMatrix { get; set; }

        public MatrixNn? ResultApplied { get; set; }

        public double ExpectedResult { get; set; } // True value for learning

        public Dictionary<int, LayerResult> LayerResults { get; }

        public NnPropagationResult(MatrixNn features)
        {
            Features = features;
            LayerResults = new Dictionary<int, LayerResult>();
        }
    }

    public class LayerResult
    {
        public int Index { get; }
        public int IndexMath { get => Index + 1; }

        public MatrixNn LayerInput { get; } // what came from previous layer
        public MatrixNn WeightsApplied { get; } // a_idx, A matrix
        public MatrixNn SigmoidApplied { get; } // z_idx, Z matrix

        public MatrixNn? HelperDelta { get; set; }
        public MatrixNn? DeltaWeight { get; set; }

        public LayerResult(int index, MatrixNn layerInput, MatrixNn weightsApplied, MatrixNn sigmoidApplied)
        {
            Index = index;
            LayerInput = layerInput;
            WeightsApplied = weightsApplied;
            SigmoidApplied = sigmoidApplied;
        }
    }


}
