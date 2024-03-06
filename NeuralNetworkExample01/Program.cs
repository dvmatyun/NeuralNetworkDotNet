// See https://aka.ms/new-console-template for more information
using NeuralNetworkExample01;
using NeuralNetworkExample01.InputOperations;
using NeuralNetworkExample01.NeuralNetworkRaw;
using System.Text;

Console.WriteLine("Program start");

//PerceptronPrograms.RunSimplePerceptron();
PerceptronPrograms.RunGamePerceptron();



class PerceptronPrograms
{
    public static void WriteExampleInput()
    {
        var inputOperation = new InputOperations();
        var exampleInputs = new List<TestInput>() { new TestInput(new MatrixNn(3, 1), new MatrixNn(2, 1)) };
        inputOperation.WriteFile(exampleInputs, path: "example.csv");
    }

    public static void RunSimplePerceptron()
    {
        Console.WriteLine("RunSimplePerceptron start");
        var nnPerceptron = new NnPerceptronSimple(3, 2, 8);
        var inputOperation = new InputOperations();
        var inputs = inputOperation.ReadFile();

        for (int i = 0; i < 100; i++)
        {
            foreach (var input in inputs)
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
    }


    public static void RunGamePerceptron()
    {
        Console.WriteLine("RunGamePerceptron start");
        var nnPerceptron = new NnPerceptronSimple(4, 2, 8);
        var inputOperation = new InputOperations();
        //var inputs = inputOperation.ReadFile();
        const double ExploreRate = 0.5;

        int turnsExpected = 40;
        for (int i = 0; i < 1000; i++)
        {
            var game = new BuildingGame();
            var state = game.State;
            var nnActions = new List<NnPropagationResult>();

            while (!state.GameEnded)
            {
                var input = state.AsInput();
                var result = nnPerceptron.ForwardPropagation(input.Input);
                var appliedResult = state.DoOutputAction(result.ResultMatrix!);
                result.ResultApplied = appliedResult;
                nnActions.Add(result);
            }

            double errorModifier = ((double)turnsExpected - (double)state.Turn + 1) / ((double)turnsExpected + (double)state.Turn);
            var sigmoidError = NnConfig.Sigmoid(errorModifier);
            if (turnsExpected >= state.Turn)
            {
                turnsExpected = state.Turn - 1;
            }

            bool isGameSuccessful = errorModifier > 0;
            double sumError = 0;
            var rand = new Random();

            foreach (var e in nnActions)
            {
                var output = e.ResultMatrix!;
                var highestActionIdx = output.FindIndexOfHighestValue();
                var diffWithActual = e.ResultApplied!.ApplyFunctionIndexed((x, _, val) => {
                    return x - output.Values[x, 0];
                });

                var expectedOutput = output.ApplyFunctionIndexed((x, _, val) =>
                {
                    var explore = rand.NextDouble() * ExploreRate;
                    if (isGameSuccessful && x == highestActionIdx)
                        return val + diffWithActual.Values[x,0];
                    if (!isGameSuccessful && x != highestActionIdx)
                        return Math.Clamp(val + explore + diffWithActual.Values[x, 0], 0, 1);
                    return val * sigmoidError + diffWithActual.Values[x, 0];
                });
                nnPerceptron.BackwardPropagation(e, expectedOutput);
                nnPerceptron.ApplyWeightsChange(e);
                sumError += Math.Abs(e.SumError);
            }

            Console.WriteLine($" > Game#{i} ended in {state.Turn} turns (sumError = {sumError}, isGameSuccessful = {isGameSuccessful})");
        }

        //var inputFinal = TestInput.ParseInput("0,2;0,2;0,4;1;0", 3, 2);
        //var resultFinal = nnPerceptron.ForwardPropagation(inputFinal.Input);
        //Console.WriteLine($"> Final prediction:");
        //resultFinal.LayerResults[1].SigmoidApplied.PrintMatrix();
    }

}