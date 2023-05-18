import _ from "underscore";
import { DataSet } from "./data/DataSet";
import { Matrix } from "./matrix/Matrix";
import { LinearRegression } from "./models/LinearRegression";
import {
  Lrelu,
  Relu,
  Tanh,
} from "./neural-networks-2/functions/ActivationFunctions";
import { CategoricalCrossEntropy } from "./neural-networks-2/functions/LossFunctions";
import { MultilayerPerceptron } from "./neural-networks-2/networks/MultilayerPerceptron";
import { GradientDescent } from "./neural-networks-2/training/GradientDescent";
import { TrainingMonitor } from "./neural-networks-2/training/TrainingMonitor";
import { GaussianDistribution } from "./prob/distributions/Gaussian";
import { UniformDistribution } from "./prob/distributions/Uniform";
import { field, maxIndex } from "./utils/helpers";
import { Vector } from "./vector/Vector";
import * as readline from "readline";

const uniform = new UniformDistribution(-0.1, 0.1);

const rl = readline.createInterface(process.stdin);

const data = new DataSet([
  {
    question: 1,
    answer: 1,
  },
  {
    question: 2,
    answer: 0,
  },
  {
    question: 3,
    answer: 1,
  },
  {
    question: 4,
    answer: 0,
  },
  {
    question: 5,
    answer: 1,
  },
  {
    question: 6,
    answer: 0,
  },
  {
    question: 7,
    answer: 1,
  },
  {
    question: 8,
    answer: 0,
  },
  {
    question: 9,
    answer: 1,
  },
  {
    question: 10,
    answer: 0,
  },
]);

const model = new MultilayerPerceptron({
  inputWidth: 2,
  hiddenWidth: [10],
  outputWidth: 2,
  hiddenActivationFuncs: [new Relu()],
  outputActivationFunc: {
    type: "softmax",
    temperature: 0.1,
  },
  parameterInitializer: () => uniform.sample(),
  normalizationLayer: false,
});

const trainer = new GradientDescent();

console.log("Training!");

trainer.train(
  model,
  data,
  (x) =>
    new Matrix([
      x.question % 2 === 0
        ? new Vector([1e-6, 1 - 1e-6])
        : new Vector([1 - 1e-6, 1e-6]),
    ]),
  (x) =>
    new Matrix([
      new Vector(
        Array.from({ length: 2 }).map((_, i) =>
          x.answer === i ? 1 - 1e-6 : 1e-6
        )
      ),
    ]),
  undefined, // new TrainingMonitor(),
  {
    learningRate: 0.01,
    stochastic: false,
    lossFunc: new CategoricalCrossEntropy(),
    batchSize: 1,
    iterations: 100000,
    /* adam: {
      momentumRatio: 0.999,
      momentumRatio2: 0.9,
    }, */
  }
);

console.log("Trained!");

const main = () =>
  rl.question(">>>", (value) => {
    const predict = model.predict(new Matrix([new Vector([parseInt(value)])]))
      .rows[0].entries;
    const z = maxIndex(predict);

    console.log(z);
    main();
  });

main();
