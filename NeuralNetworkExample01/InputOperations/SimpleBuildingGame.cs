

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

        const int FarmCost = 10;
        public int Farm { get; set; } = 0;
        public int Forester { get; set; } = 0;
        //public int StoneMine { get; set; } = 0;

        public int FoodGr { get => 2 + Farm * 2; }
        public int WoodGr { get => 4 + Forester; }
        //public int FoodGr { get => 2 + Farm * 2 - Forester - StoneMine; }

        //public int StoneGr { get => 1 + StoneMine; }

        public int Turn { get; set; } = 0;

        public bool GameEnded { get => Food > 200; }

        public void DoAction(int actionIdx)
        {
            switch (actionIdx)
            {
                case 0:
                    // Next turn:
                    Food += FoodGr;
                    Turn += 1;
                    break;
                case 1:
                    // BuildFarm:
                    Farm += 1;
                    Food -= FarmCost;
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
                output.Values[i, 0] += Math.Clamp(output.Values[i, 0] + explore, 0, 1);
            }

            if (Food < FarmCost)
            {
                output.Values[1, 0] = 0;
            }
            return output;
        }

        public TestInput AsInput()
        {
            return new TestInput(
                new MatrixNn(new double[] { Food, Farm, FoodGr, Turn }), 
                new MatrixNn(new double[] { 0, 0 }));
        }
    }

    /*
    public class BuildingGameDecision
    {
        public BuildingGameState GameState { get; set; }

        public NnPropagationResult PropagationResult { get; set; }


        public BuildingGameDecision(BuildingGameState gameState, NnPropagationResult propagationResult) 
        {
            GameState = gameState;
            PropagationResult = propagationResult;
        }
    }
    */
}
