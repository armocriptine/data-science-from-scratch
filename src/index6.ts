import _ from "underscore";
import { DataSet } from "./data/DataSet";
import { Matrix } from "./matrix/Matrix";
import { Lrelu, Relu } from "./neural-networks-2/functions/ActivationFunctions";
import { CategoricalCrossEntropy } from "./neural-networks-2/functions/LossFunctions";
import { MultilayerPerceptron } from "./neural-networks-2/networks/MultilayerPerceptron";
import { GradientDescent } from "./neural-networks-2/training/GradientDescent";
import { TrainingMonitor } from "./neural-networks-2/training/TrainingMonitor";
import { UniformDistribution } from "./prob/distributions/Uniform";
import { field, maxIndex, parseMnistDataSet } from "./utils/helpers";
import { Vector } from "./vector/Vector";

const uniform = new UniformDistribution(-0.1, 0.1);

const mlp = new MultilayerPerceptron({
  inputWidth: 784,
  hiddenWidth: [50],
  outputWidth: 10,
  hiddenActivationFuncs: [new Lrelu(0.1)],
  outputActivationFunc: {
    type: "softmax",
    temperature: 1,
  },
  parameterInitializer: () => uniform.sample(),
  normalizationLayer: true,
});

// mlp.loadParameters(process.cwd() + "/src/dataset/mnist_weights_all6_norm.json");

const mnistTrainSet = parseMnistDataSet(
  process.cwd() + "/src/dataset/mnist_train.csv"
).map((x) => ({
  features: x.features.map((y) => y / 255),
  response: x.response,
}));

const mnistTestSet = parseMnistDataSet(
  process.cwd() + "/src/dataset/mnist_test.csv"
).map((x) => ({
  features: x.features.map((y) => y / 255),
  response: x.response,
}));

const trainingSet = new DataSet(mnistTrainSet);
const validationSet = new DataSet(mnistTestSet);

const trainer = new GradientDescent();
// const monitor = new TrainingMonitor(validationSet);

console.log("TRAINING...");

trainer.train(
  mlp,
  trainingSet,
  (x) => new Matrix([new Vector(x.features)]),
  (x) =>
    new Matrix([
      new Vector(
        Array.from({ length: 10 }).map((_, i) =>
          x.response === i ? 1 - 1e-6 : 1e-6
        )
      ),
    ]),
  undefined, //monitor,
  {
    learningRate: 0.001,
    stochastic: true,
    lossFunc: new CategoricalCrossEntropy(),
    batchSize: 1,
    iterations: 1000000,
    validationSet,
    /* momentum: {
      ratio: 0.8,
    }, */
    adam: {
      momentumRatio: 0.999,
      momentumRatio2: 0.9,
    },
  }
);

console.log("Trained!");
/*

let correct = 0;

for (const instance of _.sample(mnistTestSet, 100)) {
  const pred = mlp.predict(new Matrix([new Vector(instance.features)])).rows[0];
  const z = maxIndex(pred.entries);

  console.log(pred.pretty);

  if (z === instance.response) {
    correct++;
    console.log("CORRECT!");
  } else {
    console.log("WRONG!");
  }
}

console.log({ correct });
/*

getPixels(process.cwd() + "/src/dataset/test_image.jpg", (err, pixels) => {
  const image = new Array(784);

  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      image[i * 28 + j] = Math.round(
        0.2989 * pixels.get(j, i, 0) +
          0.587 * pixels.get(j, i, 1) +
          0.114 * pixels.get(j, i, 2)
      );
    }
  }

  const predicted = mlp.predict(new Vector(image.map((x) => x / 255)));
  console.log(predicted.pretty);
  console.log(maxIndex(predicted.entries));
});
*/
