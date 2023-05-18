import { Matrix } from "../../matrix/Matrix";
import { I, O, square, sum } from "../../utils/helpers";
import { Vector } from "../../vector/Vector";

export abstract class LossFunc {
  public abstract evaluate(predicted: Matrix, expected: Matrix): number;

  public abstract differentiate(predicted: Matrix, expected: Matrix): Matrix;
}

export class MeanSquaredError extends LossFunc {
  public evaluate(predicteds: Matrix, expecteds: Matrix): number {
    return predicteds.rows
      .flatMap((row, i) =>
        row.entries.map((predicted, j) =>
          square(expecteds.at(i, j) - predicted)
        )
      )
      .reduce(sum, 0);
  }

  public differentiate(predicteds: Matrix, expecteds: Matrix): Matrix {
    return new Matrix(
      predicteds.rows.flatMap(
        (row, i) =>
          new Vector(
            row.entries.map(
              (predicted, j) => -2 * (expecteds.at(i, j) - predicted)
            )
          )
      )
    );
  }
}

export class BinaryCrossEntropy extends LossFunc {
  public evaluate(predicteds: Matrix, expecteds: Matrix): number {
    return predicteds.rows
      .flatMap((row, i) =>
        row.entries.map((predicted, j) => {
          const predict = Math.max(Math.min(predicted, 0.9999), 0.0001);
          const expect = Math.max(Math.min(expecteds.at(i, j), 0.9999), 0.0001);
          return (
            -expect * Math.log(predict) - (1 - expect) * Math.log(1 - predict)
          );
        })
      )
      .reduce(sum, 0);
  }

  public differentiate(predicteds: Matrix, expecteds: Matrix): Matrix {
    return new Matrix(
      predicteds.rows.flatMap(
        (row, i) =>
          new Vector(
            row.entries.map((predicted, j) => {
              const predict = Math.max(Math.min(predicted, 0.9999), 0.0001);
              const expect = Math.max(
                Math.min(expecteds.at(i, j), 0.9999),
                0.0001
              );
              return -(expect / predict) + (1 - expect) / (1 - predict);
            })
          )
      )
    );
  }
}

export class CategoricalCrossEntropy extends LossFunc {
  public evaluate(predicteds: Matrix, expecteds: Matrix): number {
    return expecteds.rows
      .flatMap((row, i) =>
        row.entries.map((expected, j) => {
          const predict = Math.max(Math.min(predicteds.at(i, j), I), O);
          const expect = Math.max(Math.min(expected, I), O);
          return -expect * Math.log(predict);
        })
      )
      .reduce(sum, 0);
  }

  public differentiate(predicteds: Matrix, expecteds: Matrix): Matrix {
    return new Matrix(
      expecteds.rows.flatMap(
        (row, i) =>
          new Vector(
            row.entries.map((expected, j) => {
              const predict = Math.max(Math.min(predicteds.at(i, j), I), O);
              const expect = Math.max(Math.min(expected, I), O);
              return -(expect / predict);
            })
          )
      )
    );
  }
}
