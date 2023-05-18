import { Matrix } from "../matrix/Matrix";
import { Distribution } from "../prob/distributions/Distribution";
import { SampleSpace } from "../prob/sample-spaces/SampleSpace";
import { square, sum } from "../utils/helpers";

export class Vector {
  constructor(public readonly entries: readonly number[]) {}

  public static from(dist: Distribution, count: number): Vector {
    return new Vector(new Array(count).fill(0).map(() => dist.sample()));
  }

  public static repeat(x: number, count: number): Vector {
    return new Vector(new Array(count).fill(0).map(() => x));
  }

  public static zero(count: number): Vector {
    return Vector.repeat(0, count);
  }

  public padEnd(x: number, count: number): Vector {
    return new Vector([
      ...this.entries,
      ...new Array(count - this.count).fill(0),
    ]);
  }

  public get count(): number {
    return this.entries.length;
  }

  public get pretty(): string {
    return this.entries.map((x) => x.toFixed(2)).join("\t");
  }

  public get euclideanNorm(): number {
    return Math.sqrt(this.entries.map(square).reduce(sum));
  }

  public get mean(): number {
    return this.entries.reduce(sum) / this.entries.length;
  }

  public get median(): number {
    const sorted = [...this.entries].sort();

    if (sorted.length % 2 === 0) {
      return (
        (sorted[Math.floor(sorted.length / 2) - 1] +
          sorted[Math.floor(sorted.length / 2)]) /
        2
      );
    } else {
      return sorted[Math.floor(sorted.length / 2) - 1];
    }
  }

  public get variance(): number {
    const mean = this.mean;
    return (
      this.entries
        .map((x) => x - mean)
        .map(square)
        .reduce(sum) / this.entries.length
    );
  }

  public get stdDev(): number {
    return Math.sqrt(this.variance);
  }

  public get asRow(): Matrix {
    return Matrix.fromRows([this]);
  }

  public get asCol(): Matrix {
    return Matrix.fromCols([this]);
  }

  public get normalize(): Vector {
    return this.times(1 / this.euclideanNorm);
  }

  public toSampleSpace(): SampleSpace<number> {
    return new SampleSpace(this.entries);
  }

  public covariance(another: Vector): number {
    const thisMean = this.mean;
    const anotherMean = another.mean;

    return (
      this.entries
        .map((e, i) => (e - thisMean) * (another.entries[i] - anotherMean))
        .reduce(sum) / this.entries.length
    );
  }

  public correlation(another: Vector): number {
    return this.covariance(another) / (this.stdDev * another.stdDev);
  }

  public add(another: Vector): Vector {
    return new Vector(this.entries.map((e, i) => e + another.entries[i]));
  }

  public subtract(another: Vector): Vector {
    return this.add(another.neg);
  }

  public dot(another: Vector): number {
    return this.entries.map((e, i) => e * another.entries[i]).reduce(sum);
  }

  public times(scalar: number): Vector {
    return new Vector(this.entries.map((e) => e * scalar));
  }

  public get neg(): Vector {
    return this.times(-1);
  }

  public projectOnto(another: Vector): Vector {
    return another.times(another.dot(this) / another.dot(another));
  }

  public at(index: number): number {
    return this.entries[index];
  }
}

export const vecSum = (prev: Vector, curr: Vector) => prev.add(curr);
