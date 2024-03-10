using NeuralNetworkExample01.NeuralNetworkRaw;
using System.Text;

namespace NeuralNetworkExample01
{
    public class NnPerceptronSimple
    {
        public NnConfig Config { get; }

        private const bool ShowPrints = false;

        public NnPerceptronSimple(int inputs, int outputs, int hidden)
        {
            Config = new NnConfig(new List<int> { hidden }, inputs, outputs);
        }

        public NnPerceptronSimple(int inputs, int outputs, List<int> hiddenLayers)
        {
            Config = new NnConfig(hiddenLayers, inputs, outputs);
        }
        #region Public

        public NnPropagationResult ForwardPropagation(MatrixNn features)
        {
            var result = new NnPropagationResult(features);


            var previousSigmoid = features;
            for (int l = 0; l < Config.Layers.Count; ++l)
            {
                var layer = Config.Layers[l];
                if (layer == null)
                    throw new ArgumentException($"Layer with index={l} not found!");
                if (ShowPrints)
                {
                    Console.WriteLine($"\nSource weights matrix for layer={l}:");
                    layer.Neurons.PrintMatrix();
                    Console.WriteLine($"Input for layer={l}:");
                    previousSigmoid.PrintMatrix();
                }

                var weights = layer.NeuronsTransposed;

                // Step (F1) for forward propagation if l == 0
                // Step (F3) for forward propagation if l == 1
                var z_idx = weights.MultiplyByMatrix(previousSigmoid);

                if (ShowPrints)
                {
                    Console.WriteLine($"Output of layer #{l}:");
                    z_idx.PrintMatrix();
                }

                // Step (F2) for forward propagation  if l == 0
                // Step (F4) for forward propagation  if l == 1
                var a_idx = z_idx.ApplyFunction(NnConfig.Sigmoid);

                if (ShowPrints)
                {
                    Console.WriteLine($"Output after sigmoid of layer #{l}:");
                    a_idx.PrintMatrix();
                }

                var layerResult = new LayerResult(l, previousSigmoid, z_idx, a_idx);
                result.LayerResults[l] = layerResult;
                previousSigmoid = a_idx;
            }

            result.ResultMatrix = previousSigmoid;
            return result;
        }

        public NnPropagationResult ForwardAndBackwardPropagation(MatrixNn features, MatrixNn expectedOutput, double learningRate = 1)
        {
            var forwardResult = ForwardPropagation(features);
            var backwardResult = BackwardPropagation(forwardResult, expectedOutput);
            ApplyWeightsChange(backwardResult, learningRate);
            return backwardResult;
        }

        public NnPropagationResult BackwardPropagation(NnPropagationResult result, MatrixNn expectedOutput)
        {
            var lastLayerIndex = result.LayerResults.Count - 1;
            for (int l = lastLayerIndex; l >= 0; --l)
            {
                var layer = Config.Layers[l];
                if (layer == null)
                    throw new ArgumentException($"Layer with index={l} not found!");

                var layerResult = result.LayerResults[l];
                if (layerResult == null)
                    throw new ArgumentException($"Layer result with index={l} not found!");

                // Step (B1)
                var sigmaDeriv = layerResult.WeightsApplied.ApplyFunction(NnConfig.SigmoidDerivative);

                MatrixNn? helperDelta;
                if (l == lastLayerIndex)
                {
                    if (ShowPrints)
                    {
                        Console.WriteLine("\n Sigma deriv of last layer:");
                        sigmaDeriv.PrintMatrix();
                        Console.WriteLine("Layer input was:");
                        layerResult.LayerInput.PrintMatrix();
                    }

                    // Step (B2.1)
                    helperDelta = layerResult.SigmoidApplied.SubstractMatrixes(expectedOutput);
                    double error = 0;

                    // Step (B3.1)
                    for (int i = 0; i < helperDelta.Rows; i++)
                    {
                        error += helperDelta.Values[i, 0];
                        helperDelta.Values[i, 0] = helperDelta.Values[i, 0] * sigmaDeriv.Values[i, 0];
                    }

                    result.SumError = error;
                    if (ShowPrints)
                    {
                        Console.WriteLine($"Error = {error}\n");
                    }
                }
                else
                {
                    var layerResultNext = result.LayerResults[l + 1];
                    if (layerResultNext == null)
                        throw new ArgumentException($"Layer result with index={l + 1} not found!");
                    var layerNext = Config.Layers[l + 1];
                    if (layerNext == null)
                        throw new ArgumentException($"Layer (next) with index={l + 1} not found!");

                    MatrixNn? d_1;
                    var helperDeltaPrevious = layerResultNext.HelperDelta;

                    // Step (B2.2)
                    d_1 = layerNext.Neurons.MultiplyByMatrix(helperDeltaPrevious!);

                    // Step (B3.2)
                    var multipliedD2 = MultiplyRowsByValues(d_1.Values, sigmaDeriv.Values);
                    var multipliedMatrix = new MatrixNn(multipliedD2);
                    helperDelta = multipliedMatrix;
                    
                }
                layerResult.HelperDelta = helperDelta;

                // Step (B4) Weights differencial:
                var dWeight = helperDelta.MultiplyByMatrix(layerResult.LayerInput.Transpose());
                layerResult.DeltaWeight = dWeight.Transpose();
                if (ShowPrints)
                {
                    Console.WriteLine($"Layers #{l} change:");
                    layerResult.DeltaWeight.PrintMatrix();
                }
                
            }
            //NnConfig.SigmoidDerivative
            return result;
        }

        public void ApplyWeightsChange(NnPropagationResult result, double learningRate = 1)
        {
            for (int i = 0; i < result.LayerResults.Count; ++i)
            {
                var layer = Config.Layers[i];
                if (layer == null)
                    throw new ArgumentException($"Layer with index={i} not found!");
                var layerResult = result.LayerResults[i];
                if (layerResult == null)
                    throw new ArgumentException($"Layer result with index={i} not found!");

                // Step (B5) correcting weights
                layer.CorrectWeights(layerResult.DeltaWeight!, learningRate);
                //var newWeights = 
            }
        }

        /*
        public NnPropagationResult ForwardPropagation(double[] features, double expectedValue = 0, bool doBackwards = false)
        {
            
            var firstLayer = MultiplyMatrixByVector(InputWeights, features); // z (2)
            var appliedFirst = ApplySigmoid(firstLayer); // a(2) = g( z(2) )

            var appliedHidden = MultiplyVectors(HiddenWeights, appliedFirst); // z (3)
            var resultSigmoid = NnConfig.Sigmoid(appliedHidden); // a(3) = g( z(3) )

            var result = new NnPropagationResult()
            {
                Result = resultSigmoid,
                ExpectedResult = expectedValue,
            };
            result.LayerResults[0] = new LayerResult(0, firstLayer, appliedFirst);
            result.LayerResults[1] = new LayerResult(1, new double[] { appliedHidden }, new double[] { resultSigmoid });

            for (int i = 1; i >= 0; --i)
            {

            }

            return result;
        }
        */

        public string GenerateExampleString()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < Config.Inputs; ++i)
                sb.Append($"input_{i},");
            for (int i = 0; i < Config.Outputs; ++i)
            {
                if (i == (Config.Outputs - 1))
                    sb.Append($"output_{i}");
                else
                    sb.Append($"output_{i},");
            }
            return sb.ToString();
        }
        #endregion

        #region BackwardPropagation


        public double LossFunction(double prediction, double realValue)
        {
            var loss = -1 * (realValue * Math.Log(prediction) + (1 - realValue) * Math.Log(1 - prediction));
            return loss;
        }

        /*
        private double[] HelperDeltaBackwards(int index, NnPropagationResult forward)
        {
            var layer = forward.LayerResults[index];
            var derivativeZ = ApplySigmoidDerivative(layer.VectorAfterWeights);
            if (layer.Index == 1)
            {
                
                var delta = (layer.VectorAfterSigmoid[0] - forward.ExpectedResult) * derivativeZ[0];
                layer.HelperDelta = new double[] { delta };
                return layer.HelperDelta;
            }
            else
            {
                var matrixMult = MultiplyVectorByValue(la)
            }
        }
        */



        #endregion

        #region Private Basic

       private double[,] MultiplyRowsByValues(double[,] matrix, double[,] array)
       {
            var rows = matrix.GetLength(0);
            var cols = matrix.GetLength(1);
            var newMatrix = new double[rows, cols];
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    newMatrix[i,j] = matrix[i,j] * array[i, 0];
                }
            }
            return newMatrix;
       }

        private double[] MultiplyMatrixByVector(double[,] matrix, double[] array)
        {
            var matrixRows = matrix.GetLength(0);
            var matrixCols = matrix.GetLength(1);
            var result = new double[matrixRows];

            for (int j = 0; j < matrixCols; ++j)
            {
                double multiply = 0;
                for (int i = 0; i < matrixRows; ++i)
                {
                    multiply += array[i] * matrix[j, i];
                }
                result[j] = multiply;
            }
            
            return result;
        }

        private double[] MultiplyVectorByValue(double[] vector, double value)
        {
            var result = new double[vector.Length];

            for (int i = 0; i < vector.Length; ++i)
            {
                result[i] = vector[i] * value;
            }

            return result;
        }


        private double MultiplyVectors(double[] vec1, double[] vec2)
        {
            double result = 0;
            for (int i =0;i < vec1.Length; ++i)
            {
                result += vec1[i] * vec2[i];
            }
            return result;
        }



        private double[] ApplySigmoid(double[] source)
        {
            var applied = new double[source.Length];
            for (var i = 0; i < applied.Length; ++i)
                applied[i] = NnConfig.Sigmoid(source[i]);
            return applied;
        }

        private double[] ApplySigmoidDerivative(double[] source)
        {
            var applied = new double[source.Length];
            for (var i = 0; i < applied.Length; ++i)
                applied[i] = NnConfig.SigmoidDerivative(source[i]);
            return applied;
        }



        #endregion

        
    }
}

