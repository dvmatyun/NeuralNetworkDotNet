

using NeuralNetworkExample01.NeuralNetworkRaw;

namespace NeuralNetworkExample01.InputOperations
{
    public class BuildingGame
    {
        public BuildingGameState State { get; }

        public BuildingGame()
        {
            State = new BuildingGameState();
        }
    }

    public class BuildingGameState
    {
        public int Food { get; set; } = 10;
        public int Wood { get; set; } = 0;
        //public int Stone { get; set; } = 0;

        const int FarmCostFood = 10;
        const int FarmCostWood = 10;
        public int Farm { get; set; } = 0;


        const int ForesterCostFood = 10;
        public int Forester { get; set; } = 0;
        //public int StoneMine { get; set; } = 0;

        public int FoodGr { get => 2 + Farm * 2; }


        public int WoodGr { get => Forester; }
        //public int FoodGr { get => 2 + Farm * 2 - Forester - StoneMine; }

        //public int StoneGr { get => 1 + StoneMine; }

        public int Turn { get; set; } = 0;

        public bool GameEnded { get => Food > 1000 || Turn > 300; }

        public Dictionary<int, int> ActionsDone { get; } = new Dictionary<int, int>();

        public BuildingGameState()
        {
            for (int i = 0; i < 100; ++i)
            {
                ActionsDone[i] = 0;
            }
        }

        public void RecordAction(int idx)
        {
            if (ActionsDone.ContainsKey(idx))
                ActionsDone[idx] += 1;
            else
                ActionsDone[idx] = 1;
        }
        public void DoAction(int actionIdx)
        {
            RecordAction(actionIdx);
            switch (actionIdx)
            {
                case 0:
                    // Next turn:
                    Food += FoodGr;
                    Wood += WoodGr;
                    Turn += 1;
                    break;
                case 1:
                    // BuildFarm:
                    Farm += 1;
                    Food -= FarmCostFood;
                    Wood -= FarmCostWood;
                    break;
                case 2:
                    Forester += 1;
                    Food -= ForesterCostFood;
                    break;
                case 3:
                    // Empty bad action:
                    Food = 0;
                    Wood = 0;
                    Turn += 1;
                    break;
            }
        }

        public MatrixNn DoOutputAction(MatrixNn output)
        {
            var cloned = output.ApplyFunction((e) => e);
            var enforces = EnforceRules(cloned, 0.1);
            // Detecting action with maximum weight:
            var highestActionIdx = enforces.FindIndexOfHighestValue();
            DoAction(highestActionIdx);
            return enforces;
        }

        public MatrixNn EnforceRules(MatrixNn output, double exploreRate)
        {
            var rand = new Random();
            
            for (int i = 0; i < output.Rows; i++)
            {
                var explore = (rand.NextDouble() - 0.5) * exploreRate;
                output.Values[i, 0] = Math.Clamp(output.Values[i, 0] + explore, 0, 1);
            }

            if (Food < FarmCostFood || Wood < FarmCostWood)
            {
                output.Values[1, 0] = 0;
            }
            if (Food < ForesterCostFood)
            {
                output.Values[2, 0] = 0;
            }
            return output;
        }

        public TestInput AsInput()
        {
            return new TestInput(
                new MatrixNn(new double[] { Food, Farm, FoodGr, Wood, Forester, WoodGr, Turn }), 
                new MatrixNn(new double[] { 0, 0, 0, 0 }));
        }
    }



    public class BuildingGameNeuro
    {
        public NnPerceptronSimple Perceptron { get; }

        public BuildingGame Game { get; }

        public BuildingGameNeuro(NnPerceptronSimple perceptron, BuildingGame game)
        {
            Perceptron = perceptron;
            Game = game;
        }

        public static BuildingGameNeuro RunGame(NnPerceptronSimple nnPerceptron, int turnsExpected)
        {
            var game = new BuildingGame();
            var state = game.State;
            var nnActions = new List<NnPropagationResult>();
            const double ExploreRate = 0.3;

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
                /*
                var diffWithActual = e.ResultApplied!.ApplyFunctionIndexed((x, _, val) => {
                    return val - output.Values[x, 0];
                });
                */

                var expectedOutput = output.ApplyFunctionIndexed((x, _, val) =>
                {
                    /*
                    if (x == 3)
                    {
                        return 0;
                    }
                    */
                    if (x == highestActionIdx)
                    {
                        if (isGameSuccessful)
                            return val;// + diffWithActual.Values[x, 0];
                        else
                            return val * sigmoidError;
                    }

                    if (!isGameSuccessful && x != highestActionIdx)
                    {
                        var explore = rand.NextDouble() * sigmoidError * ExploreRate;
                        return Math.Clamp(val + explore, 0, 1);// + diffWithActual.Values[x, 0], 0, 1);
                    }

                    return val;// + diffWithActual.Values[x, 0];
                });

                nnPerceptron.BackwardPropagation(e, expectedOutput);
                nnPerceptron.ApplyWeightsChange(e);
                /*
                if (isGameSuccessful)
                {
                    for (int a = 0; a < 10; ++a)
                    {
                        nnPerceptron.ApplyWeightsChange(e);
                    }
                }
                */
                sumError += Math.Abs(e.SumError);
            }

            return new BuildingGameNeuro(nnPerceptron, game);
        }
    }
}
