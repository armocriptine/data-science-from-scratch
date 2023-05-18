import _ from "underscore";
import { DataSet } from "../../data/DataSet";
import { MovingAverage } from "../../utils/MovingAverage";
import { maxIndex, sum } from "../../utils/helpers";
import { Vector } from "../../vector/Vector";
import { LossFunc } from "../functions/LossFunctions";
import { NeuralNetwork } from "../networks/NeuralNetwork";
import { Matrix } from "../../matrix/Matrix";

export type TrainingMonitorNotification<T extends object> = {
  network: NeuralNetwork;
  iteration: number;
  lossFunc: LossFunc;
  trainingSet: T[];
  predictors: (entry: T) => Matrix;
  responses: (entry: T) => Matrix;
};

export type TrainingMonitorResult = {
  stopTraining: boolean;
};

export abstract class TrainingMonitor<T extends object> {
  public averageTrainingLoss = new MovingAverage();
  public averageValidationLoss = new MovingAverage();
  public averageTrainingAccuracy = new MovingAverage();
  public averageValidationAccuracy = new MovingAverage();

  constructor(public readonly validationSet?: DataSet<T>) {}

  public notify(input: TrainingMonitorNotification<T>): TrainingMonitorResult {
    this.averageTrainingLoss.push(this.calculateTrainingLoss(input));

    if (this.validationSet) {
      this.averageValidationLoss.push(this.calculateValidationLoss(input));
      this.averageValidationAccuracy.push(
        this.calculateValidationAccuracy(input)
      );
    }

    this.averageTrainingAccuracy.push(this.calculateTrainingAccuracy(input));

    console.log({
      iteration: input.iteration,
      trainingLoss: this.averageTrainingLoss.average,
      trainingAccuracy: this.averageTrainingAccuracy.average,
      validationLoss: this.validationSet
        ? this.averageValidationLoss.average
        : null,
      validationAccuracy: this.averageValidationAccuracy.average,
    });

    return {
      stopTraining: false,
    };
  }

  protected calculateTrainingLoss(
    input: TrainingMonitorNotification<T>
  ): number {
    return (
      input.trainingSet
        .map((instance) => {
          const predicteds = input.network.predict(input.predictors(instance));

          const expecteds = input.responses(instance);

          return input.lossFunc.evaluate(predicteds, expecteds);
        })
        .reduce(sum, 0) / input.trainingSet.length
    );
  }

  private calculateValidationLoss(
    input: TrainingMonitorNotification<T>
  ): number {
    const instance = this.validationSet!.sample(1).entries[0];

    const predicteds = input.network.predict(input.predictors(instance));

    const expecteds = input.responses(instance);

    return input.lossFunc.evaluate(predicteds, expecteds);
  }

  protected abstract calculateValidationAccuracy(
    input: TrainingMonitorNotification<T>
  ): number;

  protected abstract calculateTrainingAccuracy(
    input: TrainingMonitorNotification<T>
  ): number;
}
