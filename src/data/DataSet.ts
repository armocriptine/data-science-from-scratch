import _ from "underscore";
import { Matrix } from "../matrix/Matrix";
import { MatrixWithHeader } from "../matrix/MatrixWithHeader";
import { Vector } from "../vector/Vector";

export class DataSet<T extends object> {
  constructor(public readonly entries: readonly T[]) {}

  public column(key: keyof T): Vector {
    return new Vector(
      this.entries
        .map((e) => e[key] as number)
        .filter((x): x is number => typeof x === "number")
    );
  }

  public get numericColumns(): (keyof T)[] {
    const entry = this.entries[0];
    const keys = Object.keys(entry).filter(
      (k) => typeof entry[k as keyof T] === "number"
    ) as (keyof T)[];

    return keys;
  }

  public toMatrix(
    rowHeaders: (entry: T, index: number) => string = (_, index) =>
      index.toString()
  ): MatrixWithHeader {
    const columns = this.numericColumns;

    return MatrixWithHeader.fromColsWithHeader(
      columns.map((c) => this.column(c)),
      this.entries.map(rowHeaders),
      columns as string[]
    );
  }

  public get covarianceMatrix(): MatrixWithHeader {
    const columns = this.numericColumns;

    return MatrixWithHeader.fromColsWithHeader(
      columns.map(
        (k1) =>
          new Vector(
            columns.map((k2) => this.column(k1).covariance(this.column(k2)))
          )
      ),
      columns as string[],
      columns as string[]
    );
  }

  public get correlationMatrix(): MatrixWithHeader {
    const columns = this.numericColumns;

    return new MatrixWithHeader(
      columns.map(
        (k1) =>
          new Vector(
            columns.map((k2) => this.column(k1).covariance(this.column(k2)))
          )
      ),
      columns as string[],
      columns as string[]
    );
  }

  public sample(count: number): DataSet<T> {
    return new DataSet(_.sample(this.entries, count));
  }
}
