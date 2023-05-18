import { readFileSync } from "fs";
import {
  embedLatinCharacters,
  embedThaiCharacters,
  unembedLatinCharacters,
} from "./nlp/embed-thai-characters";
import { Transformer } from "./neural-networks-2/networks/Transformer";
import { Relu } from "./neural-networks-2/functions/ActivationFunctions";
import { DataSet } from "./data/DataSet";
import { GradientDescent } from "./neural-networks-2/training/GradientDescent";
import {
  TrainingMonitor,
  TrainingMonitorNotification,
  TrainingMonitorResult,
} from "./neural-networks-2/training/TrainingMonitor";
import { CategoricalCrossEntropy } from "./neural-networks-2/functions/LossFunctions";
import readline from "readline";
import { Matrix } from "./matrix/Matrix";
import _ from "underscore";
import { maxIndex, sum } from "./utils/helpers";
import { UniformDistribution } from "./prob/distributions/Uniform";
import { Vector } from "./vector/Vector";
import { AttentionLayer } from "./neural-networks-2/layers/Attention";
import { MatrixLayer } from "./neural-networks-2/layers/Matrix";
import { ConstantNode } from "./neural-networks-2/nodes/Constant";
import { SimpleLayer } from "./neural-networks-2/layers/Layer";
import { DecoderBlock } from "./neural-networks-2/layers/DecoderBlock";

const rl = readline.createInterface(process.stdin);

const O = 1e-8;
const I = 1 - O;

const dataset = [
  {
    feature: Matrix.fromRaw([
      [I, O, O],
      [O, I, O],
      [O, O, I],
    ]),
    response: Matrix.fromRaw([
      [O, O, I],
      [O, I, O],
      [I, O, O],
      [O, I, O],
      [O, O, I],
    ]),
  },
  {
    feature: Matrix.fromRaw([
      [O, O, I],
      [O, I, O],
      [I, O, O],
    ]),
    response: Matrix.fromRaw([
      [I, O, O],
      [O, I, O],
      [O, O, I],
      [O, I, O],
      [I, O, O],
    ]),
  },
];

const trainSet = new DataSet(dataset);
const validationSet = new DataSet(dataset);

const transformer = new Transformer({
  encoder: {
    inputWidth: 3,
    inputLength: 3,
    count: 1,
    attentionHeads: 1,
    softmaxTemp: 0.2,
    projectedKeyQuerySize: 3,
    projectedValueSize: 3,
    feedforwardSize: 10,
    feedforwardActivationFunc: new Relu(),
    dropoutRate: 0.2,
  },
  decoder: {
    inputWidth: 3,
    inputLength: 5,
    count: 1,
    attentionHeads: 1,
    softmaxTemp: 0.2,
    projectedKeyQuerySize: 3,
    projectedValueSize: 3,
    feedforwardSize: 10,
    feedforwardActivationFunc: new Relu(),
    outputTemp: 1,
    dropoutRate: 0.2,
  },
});

transformer.loadParameters(
  process.cwd() + "/src/dataset/mnist_weights_all6_norm.json"
);

/*
transformer.saveParameters(
  process.cwd() + "/src/dataset/mnist_weights_all6_norm.json"
);
*/
/*
export class Monitor extends TrainingMonitor<(typeof dataset)[0]> {
  public override notify(
    input: TrainingMonitorNotification<(typeof dataset)[0]>
  ): TrainingMonitorResult {
    input.network.saveParameters(
      process.cwd() + "/src/dataset/mnist_weights_all6_norm.json"
    );

    super.notify(input);

    return {
      stopTraining:
        this.averageTrainingAccuracy.count > 500 &&
        this.averageTrainingAccuracy.average > 0.995,
    };
  }

  protected calculateValidationAccuracy(
    input: TrainingMonitorNotification<(typeof dataset)[0]>
  ): number {
    const instances = this.validationSet!.sample(1).entries;

    const predicteds = instances.map((instance) =>
      input.network.predict(input.predictors(instance))
    );

    const expecteds = instances.map((instance) => input.responses(instance));

    return (
      predicteds
        .map(
          (predicted, i) =>
            predicted.rows.filter(
              (predictedRow, j) =>
                maxIndex(predictedRow.entries) ===
                maxIndex(expecteds[i].rows[j].entries)
            ).length / predicted.height
        )
        .reduce(sum) / predicteds.length
    );
  }

  protected calculateTrainingAccuracy(
    input: TrainingMonitorNotification<(typeof dataset)[0]>
  ): number {
    const instances = _.sample(input.trainingSet, 1);

    const predicteds = instances.map((instance) =>
      input.network.predict(input.predictors(instance))
    );

    const expecteds = instances.map((instance) => input.responses(instance));

    return (
      predicteds
        .map(
          (predicted, i) =>
            predicted.rows.filter(
              (predictedRow, j) =>
                maxIndex(predictedRow.entries) ===
                maxIndex(expecteds[i].rows[j].entries)
            ).length / predicted.height
        )
        .reduce(sum) / predicteds.length
    );
  }
}

console.log("Training...");

const trainer = new GradientDescent();
trainer.train(
  transformer,
  trainSet,
  (x) =>
    x.feature
      .concatRows(Matrix.fromRaw([[O, O, O]]))
      .concatRows(Matrix.fromRows(x.response.rows.slice(0, 4))),
  (x) => x.response,
  new Monitor(validationSet),
  {
    learningRate: 0.001,
    stochastic: true,
    lossFunc: new CategoricalCrossEntropy(),
    batchSize: 1,
    iterations: 1000000,
    validationSet,
    /* momentum: {
      ratio: 0.8,
    }, 
    adam: {
      momentumRatio: 0.999,
      momentumRatio2: 0.9,
    },
  }
);

console.log("Trained!");
*/

const x = transformer.predict(
  Matrix.fromRaw([
    [O, O, I],
    [O, I, O],
    [I, O, O],
  ]).concatRows(
    Matrix.fromRaw([
      [O, O, O],
      [I, O, O],
      [O, I, O],
      [O, O, O],
      [O, O, O],
    ])
  )
);

console.log(x.pretty);
