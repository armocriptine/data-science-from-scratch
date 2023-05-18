import { readFileSync } from "fs";
import {
  embedLatinCharacters,
  embedThaiCharacters,
  unembedLatinCharacters,
  unembedThaiCharacters,
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
import _ from "underscore";
import { maxIndex, sum } from "./utils/helpers";
import { MatMulLayer } from "./neural-networks-2/layers/MatMul";
import { MatrixLayer } from "./neural-networks-2/layers/Matrix";
import { UniformDistribution } from "./prob/distributions/Uniform";
import { ParameterLayer } from "./neural-networks-2/layers/Parameter";
import { Vector } from "./vector/Vector";

const rl = readline.createInterface(process.stdin);

const rand = new UniformDistribution(-1, 1);

const dataset = readFileSync(process.cwd() + "/src/dataset/thai_romanize.txt")
  .toString()
  .split("\n")
  .map((l) => l.split("\t"))
  .map(([thai, latin]) => ({
    thai,
    latin,
  }))
  .filter(
    ({ thai, latin }) => thai && latin && thai.length <= 3 && latin.length <= 3
  );

const trainSet = new DataSet(dataset);
const validationSet = new DataSet(dataset);

const transformer = new Transformer({
  encoder: {
    inputWidth: 11,
    inputLength: 3,
    count: 4,
    attentionHeads: 6,
    softmaxTemp: 1,
    projectedKeyQuerySize: 10,
    projectedValueSize: 10,
    feedforwardSize: 30,
    feedforwardActivationFunc: new Relu(),
    dropoutRate: 0.2,
  },
  decoder: {
    inputWidth: 11,
    inputLength: 4,
    count: 4,
    attentionHeads: 6,
    softmaxTemp: 1,
    projectedKeyQuerySize: 10,
    projectedValueSize: 10,
    feedforwardSize: 30,
    feedforwardActivationFunc: new Relu(),
    outputTemp: 1,
    dropoutRate: 0.2,
    unembedder: (input) =>
      new MatMulLayer(
        input,
        new MatrixLayer({
          cols: Array.from({ length: 30 }).map(
            () => new ParameterLayer(11, Vector.from(rand, 11))
          ),
        })
      ),
  },
});

transformer.loadParameters(process.cwd() + "/src/dataset/transformer2.json");
/*
console.log("Training...");

export class Monitor extends TrainingMonitor<(typeof dataset)[0]> {
  public override notify(
    input: TrainingMonitorNotification<(typeof dataset)[0]>
  ): TrainingMonitorResult {
    input.network.saveParameters(
      process.cwd() + "/src/dataset/transformer2.json"
    );

    super.notify(input);

    return {
      stopTraining:
        input.iteration > 200 && this.averageValidationAccuracy.average > 0.9,
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
      expecteds.filter((expected, i) => {
        const predictedText = unembedLatinCharacters(predicteds[i]);
        const expectedText = unembedLatinCharacters(expected);

        return expectedText === predictedText;
      }).length / expecteds.length
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
      expecteds.filter((expected, i) => {
        const feature = unembedThaiCharacters(
          embedThaiCharacters(instances[i].thai, false, 3)
        );
        const predictedText = unembedLatinCharacters(predicteds[i], false);
        const expectedText = unembedLatinCharacters(expected, false);

        console.log({
          feature,
          predictedText,
          expectedText,
        });

        return expectedText === predictedText;
      }).length / expecteds.length
    );
  }
}

const trainer = new GradientDescent();
trainer.train(
  transformer,
  trainSet,
  (x) =>
    embedThaiCharacters(x.thai, true, 3).concatRows(
      embedLatinCharacters(x.latin, true, true, 4)
    ),
  (x) => embedLatinCharacters(x.latin, false, false, 4, false),
  new Monitor(validationSet),
  {
    learningRate: 0.001,
    stochastic: true,
    lossFunc: new CategoricalCrossEntropy(),
    batchSize: 10,
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

const main = () => {
  rl.question(">>", (text) => {
    let result = ''

    for (let i = 0; i < 3; i++) {
      const input = embedThaiCharacters(text, true, 3).concatRows(
        embedLatinCharacters(result, true, true, 4)
      );

      const res = transformer.predict(input);
      result += unembedLatinCharacters(res, false)[i];

      if (result.endsWith('$')) {
        break;
      }
    }

    console.log(result);

    main();
  });
};

main();
