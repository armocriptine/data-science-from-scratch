import _ from "underscore";
import { DataSet } from "../../data/DataSet";
import { Vector } from "../../vector/Vector";
import { LossFunc, MeanSquaredError } from "../functions/LossFunctions";
import { NeuralNetwork } from "../networks/NeuralNetwork";
import { Trainer } from "./Trainer";
import { TrainingMonitor } from "./TrainingMonitor";
import { Matrix } from "../../matrix/Matrix";
import { randomUUID } from "crypto";

export class GradientDescent extends Trainer {
  public train<T extends object>(
    network: NeuralNetwork,
    dataSet: DataSet<T>,
    predictors: (entry: T) => Matrix,
    responses: (entry: T) => Matrix,
    monitor?: TrainingMonitor<T>,
    options: {
      learningRate: number;
      stochastic: boolean;
      lossFunc: LossFunc;
      batchSize: number;
      iterations: number;
      validationSet?: DataSet<T>;
      momentum?: {
        ratio: number;
      } | null;
      adam?: {
        momentumRatio: number;
        momentumRatio2: number;
      } | null;
    } = {
      learningRate: 0.0001,
      stochastic: true,
      lossFunc: new MeanSquaredError(),
      batchSize: 1000,
      iterations: 100,
      momentum: null,
      adam: null,
    }
  ): void {
    const parameters = network.parameters.filter((p) => p.learnable);

    if (options.momentum) {
      for (const p of parameters) {
        p.additionalData = {
          lastVelocity: 0,
        } as MomentumParams;
      }
    } else if (options.adam) {
      for (const p of parameters) {
        p.additionalData = {
          accumulatedDerivative: 1e-6,
          accumulatedSquaredDerivative: 1e-6,
          decayedMomentumRatio: 1,
          decayedMomentumRatio2: 1,
        } as AdamParams;
      }
    }

    let trainSet: T[] = [];

    for (let iteration = 1; iteration <= options.iterations; iteration++) {
      if (trainSet.length === 0) trainSet = _.shuffle(dataSet.entries);

      const sampledDataSet = options.stochastic
        ? Array.from({
            length: Math.ceil(
              Math.min(options.batchSize, 0.3 * dataSet.entries.length)
            ),
          })
            .map(() => trainSet.pop())
            .filter((x): x is T => !!x)
        : trainSet;

      for (const instance of sampledDataSet) {
        const sessionId = randomUUID();

        const predicteds = network.predict(predictors(instance), sessionId, true);

        const expecteds = responses(instance);

        const lossGradient = options.lossFunc.differentiate(
          predicteds,
          expecteds
        );

        network.setLossGradient(lossGradient);

        for (const params of parameters) {
          params.prebackprop(sessionId, true);
        }
      }

      for (const parameter of parameters) {
        let step: number;

        if (options.momentum) {
          const derivative =
            parameter.accumulatedGradient / sampledDataSet.length;

          const momentumParams = parameter.additionalData as MomentumParams;

          const newVelocity =
            options.momentum.ratio * momentumParams.lastVelocity +
            -(options.learningRate * derivative);

          step = newVelocity;
          momentumParams.lastVelocity = newVelocity;
        } else if (options.adam) {
          const derivative =
            parameter.accumulatedGradient / sampledDataSet.length;

          const adamParams = parameter.additionalData as AdamParams;

          adamParams.accumulatedSquaredDerivative =
            options.adam.momentumRatio *
              adamParams.accumulatedSquaredDerivative +
            (1 - options.adam.momentumRatio) * derivative * derivative;
          adamParams.accumulatedDerivative =
            options.adam.momentumRatio2 * adamParams.accumulatedDerivative +
            (1 - options.adam.momentumRatio2) * derivative;
          adamParams.decayedMomentumRatio *= options.adam.momentumRatio;
          adamParams.decayedMomentumRatio2 *= options.adam.momentumRatio2;

          const adjustedLearningRate =
            (options.learningRate *
              Math.sqrt(1 - adamParams.decayedMomentumRatio)) /
            (1 - adamParams.decayedMomentumRatio2);

          step =
            -(adjustedLearningRate * adamParams.accumulatedDerivative) /
            Math.sqrt(adamParams.accumulatedSquaredDerivative);
        } else {
          step =
            -(options.learningRate * parameter.accumulatedGradient) /
            sampledDataSet.length;
        }

        parameter.adjustValue(step);
      }

      const res = monitor?.notify({
        network,
        iteration: iteration,
        lossFunc: options.lossFunc,
        trainingSet: sampledDataSet,
        predictors,
        responses,
      });

      if (res?.stopTraining) {
        break;
      }
    }
  }
}

export type MomentumParams = {
  lastVelocity: number;
};

export type AdamParams = {
  accumulatedSquaredDerivative: number;
  accumulatedDerivative: number;
  decayedMomentumRatio: number;
  decayedMomentumRatio2: number;
};
