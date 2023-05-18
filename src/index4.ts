import { DataSet } from "./data/DataSet";
import { Matrix } from "./matrix/Matrix";
import { Identity } from "./neural-networks-2/functions/ActivationFunctions";
import { MeanSquaredError } from "./neural-networks-2/functions/LossFunctions";
import { MultilayerPerceptron } from "./neural-networks-2/networks/MultilayerPerceptron";
import { GradientDescent } from "./neural-networks-2/training/GradientDescent";
import { TrainingMonitor } from "./neural-networks-2/training/TrainingMonitor";
import { UniformDistribution } from "./prob/distributions/Uniform";
import { Vector } from "./vector/Vector";

const uniform = new UniformDistribution(-1, 1);

const mlp = new MultilayerPerceptron({
  inputWidth: 2,
  hiddenWidth: [1],
  outputWidth: 1,
  hiddenActivationFuncs: [new Identity()],
  outputActivationFunc: new Identity(),
  parameterInitializer: () => uniform.sample(),
  normalizationLayer: false,
});

const trainingSet = new DataSet([
  {
    a1: 2,
    a2: 6,
    b: 8,
  },
  {
    a1: 8,
    a2: 2,
    b: 10,
  },
]);

const trainer = new GradientDescent();
trainer.train(
  mlp,
  trainingSet,
  (x) => new Matrix([new Vector([x.a1, x.a2])]),
  (y) => new Matrix([new Vector([y.b])]),
  undefined, //new TrainingMonitor(),
  {
    learningRate: 0.001,
    stochastic: false,
    lossFunc: new MeanSquaredError(),
    batchSize: 1000,
    iterations: 1000,
  }
);

console.log({
  prediction: mlp.predict(new Matrix([new Vector([2, 6])])).pretty,
});
