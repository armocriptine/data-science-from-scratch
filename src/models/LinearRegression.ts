import { DataSet } from "../data/DataSet";
import { Matrix } from "../matrix/Matrix";
import { UniformDistribution } from "../prob/distributions/Uniform";
import { Vector } from "../vector/Vector";
import { Model } from "./Model";

export class LinearRegression extends Model {
  public weight: Matrix | null = null;

  constructor(public readonly withIntercept: boolean) {
    super();
  }

  public train<T extends object>(
    dataSet: DataSet<T>,
    predictors: ((entry: T) => number)[],
    responses: ((entry: T) => number)[],
    options?: {
      method:
        | "normal-equation"
        | "full-gradient-descent"
        | "stochastic-gradient-descent";
      learningRate?: number;
      convergenceThreshold?: number;
    }
  ): void {
    const preds = Matrix.fromCols([
      ...(this.withIntercept
        ? [new Vector(new Array(dataSet.entries.length).fill(1))]
        : []), // bias (intercept) term
      ...predictors.map(
        (pred) => new Vector(dataSet.entries.map((entry) => pred(entry)))
      ),
    ]);
    const resps = Matrix.fromCols([
      new Vector(dataSet.entries.map((e) => responses[0](e))),
    ]);

    // console.log(preds.pretty);

    switch (options?.method) {
      case "normal-equation":
        this.trainByNormalEquation(preds, resps);
        break;
      case "full-gradient-descent":
        this.trainByGradientDescent(
          preds,
          resps,
          options.learningRate,
          options.convergenceThreshold,
          false
        );
        break;
      case "stochastic-gradient-descent":
        this.trainByGradientDescent(
          preds,
          resps,
          options.learningRate,
          options.convergenceThreshold,
          true
        );
        break;
      default:
        throw new Error("Unspecified method");
    }
  }

  public predict(predictor: Vector): number {
    if (!this.weight) throw new Error("Not trained yet!");

    return this.weight.transpose.multiply(
      new Vector([...(this.withIntercept ? [1] : []), ...predictor.entries])
        .asCol
    ).rows[0].entries[0];
  }

  private trainByNormalEquation(preds: Matrix, resps: Matrix): void {
    this.weight = preds.transpose
      .multiply(preds)
      .inverse.multiply(preds.transpose)
      .multiply(resps);
    console.log(this.weight.transpose.pretty);
  }

  private trainByGradientDescent(
    preds: Matrix,
    resps: Matrix,
    learningRate = 0.001,
    convergenceThreshold = 0.000001,
    stochastic = false
  ): void {
    const rand = new UniformDistribution(0, preds.height - 1);
    const sampleCount = Math.max(1, Math.round(preds.height * 0.3));

    let weight = Matrix.fromCols([
      Vector.from(new UniformDistribution(-1, 1), preds.width),
    ]);

    for (;;) {
      let sampledPreds = preds;
      let sampledResps = resps;

      if (stochastic) {
        const indexes = new Array(sampleCount)
          .fill(0)
          .map(() => Math.round(rand.sample()));
        sampledPreds = Matrix.fromRows(
          preds.rows.filter((_, i) => indexes.includes(i))
        );
        sampledResps = Matrix.fromRows(
          resps.rows.filter((_, i) => indexes.includes(i))
        );
      }

      const currentWeight = weight.subtract(
        sampledPreds.transpose
          .multiply(sampledPreds.multiply(weight).subtract(sampledResps))
          .times(learningRate)
      );

      console.log(weight.transpose.pretty);

      const residual = currentWeight.cols[0].subtract(
        weight.cols[0]
      ).euclideanNorm;

      if (residual <= convergenceThreshold) {
        break;
      }

      weight = currentWeight;
    }

    this.weight = weight;
  }
}
