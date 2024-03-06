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
                var result = nnPerceptron.ForwardAndBackwardPropagation(input.Input, input.Output, learningRate: 10);
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
        var nnPerceptron = new NnPerceptronSimple(7, 4, new List<int> { 4 });
        var inputOperation = new InputOperations();
        //var inputs = inputOperation.ReadFile();
        

           
        int turnsExpected = 100;
        int prevTurnResult = 100;

        int avgGameTurns = 200;

        const int iterations = 20;
        int cyclesDone = iterations;
        // Genetic algorithm
        var bestPopulation = new List<BuildingGameNeuro>();
        
        
        for (int i = 0; i < iterations; ++i)
        {
            AddNewPopulationToList(bestPopulation, iterations, turnsExpected);
            var nextGen = new List<BuildingGameNeuro>();
            foreach (var e in bestPopulation)
            {
                var gen = BuildingGameNeuro.RunGame(e.Perceptron, turnsExpected);
                nextGen.Add(gen);
            }
            bestPopulation.AddRange(nextGen);
            bestPopulation = bestPopulation.OrderBy(e => e.Game.State.Turn).ToList();
            bestPopulation = bestPopulation.Take(cyclesDone).ToList();
            var bestState = bestPopulation.First().Game.State;
            var worstState = bestPopulation.Last().Game.State;
            Console.WriteLine($"Gen #{i} Best result: {bestState.Turn} Turns (Bad actions: {bestState.ActionsDone[3]})" +
                $", worst result: {worstState.Turn} Turns (Bad actions: {worstState.ActionsDone[3]})");
            
            /*
            Mutate here:
            for (int j = 0; j < bestPopulation.Count; ++j)
            {
            }
            */
            cyclesDone -= 1;
        }

        //Console.WriteLine($" > Minimum turns game: {turnsExpected}");

        //var inputFinal = TestInput.ParseInput("0,2;0,2;0,4;1;0", 3, 2);
        //var resultFinal = nnPerceptron.ForwardPropagation(inputFinal.Input);
        //Console.WriteLine($"> Final prediction:");
        //resultFinal.LayerResults[1].SigmoidApplied.PrintMatrix();
    }

    public static void AddNewPopulationToList(List<BuildingGameNeuro> list, int iterations, int turnsExpected)
    {
        // Initial population:
        for (int i = 0; i < iterations; ++i)
        {
            var gameResult = BuildingGameNeuro.RunGame(new NnPerceptronSimple(7, 4, new List<int> { 8, 16 }), turnsExpected);
            list.Add(gameResult);
            //Console.WriteLine($" > Game#{i} ended in {gameResult.Game.State.Turn} turns");
        }
    }

}