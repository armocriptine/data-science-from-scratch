import { DataSet } from "../data/DataSet";
import { Vector } from "../vector/Vector";

export abstract class Model {
  public abstract train<T extends object>(
    dataSet: DataSet<T>,
    predictors: ((entry: T) => number)[],
    responses: ((entry: T) => number)[]
  ): void;

  public abstract predict(predictor: Vector): number | Vector;
}
